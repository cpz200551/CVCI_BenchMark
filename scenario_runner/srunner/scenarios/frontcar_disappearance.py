import py_trees
import carla
import time
import math

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, DriveDistance
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    CollisionTest,
    StaticObstacleSlowDownCriterion,
    StaticObstacleNoCollisionCriterion,
    StaticObstacleSafePassCriterion
)

class PassiveEgoSpeedHold(py_trees.behaviour.Behaviour):
    """
    保持自车初始速度，直到自动驾驶算法开始接管
    """
    def __init__(self, actor, target_speed, name="PassiveEgoSpeedHold"):
        super().__init__(name)
        self.actor = actor
        self.target_speed = target_speed
        self._manual_override_detected = False

    def update(self):
        if not self.actor:
            return py_trees.common.Status.RUNNING

        control = self.actor.get_control()
        # 判断是否还处于“无手动/算法控制”状态
        passive_input = (
            abs(control.steer) < 0.05 and
            control.brake < 0.05 and
            control.throttle < 0.05
        )

        # 一旦有控制输入，就不再接管
        if not passive_input:
            self._manual_override_detected = True

        # 没被接管时，持续设置目标速度
        if not self._manual_override_detected and passive_input:
            transform = self.actor.get_transform()
            forward = transform.get_forward_vector()
            self.actor.set_target_velocity(carla.Vector3D(
                forward.x * self.target_speed,
                forward.y * self.target_speed,
                forward.z * self.target_speed
            ))

        return py_trees.common.Status.RUNNING
        
# 自定义连续变道行为节点
class ContinuousLaneChange(AtomicBehavior):
    def __init__(self, vehicle, target_lateral_distance=3.5, angle_max=15.0, name="ContinuousLaneChange"):
        super().__init__(name)
        self.vehicle = vehicle
        self.target_lateral = target_lateral_distance
        self.current_lateral = 0.0
        self.angle_max = angle_max
        self.initial_yaw = None
        self._destroyed = False

    def update(self):
        if not self.vehicle or self._destroyed:
            return py_trees.common.Status.FAILURE

        try:
            trans = self.vehicle.get_transform()
        except RuntimeError:
            self._destroyed = True
            return py_trees.common.Status.FAILURE

        if self.initial_yaw is None:
            self.initial_yaw = trans.rotation.yaw

        step = 0.15
        remaining = self.target_lateral - self.current_lateral
        if remaining < 0.05:
            final_rot = carla.Rotation(pitch=0, yaw=self.initial_yaw, roll=0)
            self.vehicle.set_transform(carla.Transform(trans.location, final_rot))
            return py_trees.common.Status.SUCCESS

        move = min(step, remaining)
        self.current_lateral += move

        yaw_rad = math.radians(self.initial_yaw)
        dx = move * math.cos(yaw_rad - math.pi/2)
        dy = move * math.sin(yaw_rad - math.pi/2)

        new_loc = trans.location
        new_loc.x += dx
        new_loc.y += dy

        ratio = self.current_lateral / self.target_lateral
        if ratio < 0.5:
            angle = -self.angle_max * (ratio * 2)
        else:
            angle = -self.angle_max * ((1.0 - ratio) * 2)

        new_yaw = self.initial_yaw + angle
        new_rot = carla.Rotation(pitch=0, yaw=new_yaw, roll=0)

        self.vehicle.set_transform(carla.Transform(new_loc, new_rot))
        time.sleep(0.05)
        return py_trees.common.Status.RUNNING

# 自定义前车行驶行为
class LeadVehicleDrive(AtomicBehavior):
    def __init__(self, lead_vehicle, ego_vehicle, speed=30.0, name="LeadVehicleDrive"):
        super().__init__(name)
        self.lead_vehicle = lead_vehicle
        self.ego_vehicle = ego_vehicle
        self.speed = speed / 3.6
        self.dt = 0.05
        self._destroyed = False
        self.ego_initial_location = None
        self.ego_moved = False

    def initialise(self):
        try:
            self.ego_initial_location = self.ego_vehicle.get_location()
        except RuntimeError:
            self.ego_initial_location = None

    def update(self):
        if not self.lead_vehicle or self._destroyed:
            return py_trees.common.Status.FAILURE

        if not self.ego_moved:
            if self.ego_initial_location is None:
                return py_trees.common.Status.RUNNING
            try:
                current_ego_loc = self.ego_vehicle.get_location()
                distance = current_ego_loc.distance(self.ego_initial_location)
                if distance > 0.5:
                    self.ego_moved = True
                else:
                    return py_trees.common.Status.RUNNING
            except RuntimeError:
                return py_trees.common.Status.RUNNING

        try:
            current_transform = self.lead_vehicle.get_transform()
            self.lead_vehicle.apply_control(carla.VehicleControl(hand_brake=False, brake=0.0, throttle=0.5))

            distance_per_frame = self.speed * self.dt
            yaw_rad = math.radians(current_transform.rotation.yaw)

            new_location = current_transform.location
            new_location.x += distance_per_frame * math.cos(yaw_rad)
            new_location.y += distance_per_frame * math.sin(yaw_rad)

            new_transform = carla.Transform(new_location, current_transform.rotation)
            self.lead_vehicle.set_transform(new_transform)

            return py_trees.common.Status.RUNNING

        except Exception:
            self._destroyed = True
            return py_trees.common.Status.FAILURE

class CarDisappearDiagonalAccident(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=120):
        self.timeout = timeout
        self.lead_vehicle = None
        self.accident_vehicle = None
        self.lane_change_total_distance = 3.5
        self.lane_change_angle_max = 15.0
        self.ego_vehicle = ego_vehicles[0]
        self.lead_vehicle_speed = 30.0
        self.trigger_distance_ego_lead = 20.0
        self.accident_vehicle_x = 300.2
        self.lead_vehicle_initial_x = 268.2
        self.global_y = 45.1
        self.ego_initial_speed = 9.0
        super().__init__("CarDisappearDiagonalAccident",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        world = CarlaDataProvider.get_world()
        ego_initial_transform = self.ego_vehicle.get_transform()
        ego_initial_transform.location.y = self.global_y
        self.ego_vehicle.set_transform(ego_initial_transform)
        ego_yaw = ego_initial_transform.rotation.yaw

        # 生成故障车
        if len(config.other_actors) > 0:
            accident_actor_conf = config.other_actors[0]
            spawn_transform = accident_actor_conf.transform
            spawn_transform.location.x = self.accident_vehicle_x
            spawn_transform.location.y = self.global_y
            spawn_transform.rotation.yaw = 0.0
            self.accident_vehicle = CarlaDataProvider.request_new_actor(accident_actor_conf.model, spawn_transform)
            if self.accident_vehicle:
                light_state = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam | carla.VehicleLightState.Special1
                self.accident_vehicle.set_light_state(carla.VehicleLightState(light_state))
                self.accident_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                self.other_actors.append(self.accident_vehicle)

        # 生成前车
        lead_spawn_transform = carla.Transform(
            carla.Location(x=self.lead_vehicle_initial_x, y=self.global_y, z=0.5),
            carla.Rotation(pitch=0.0, yaw=ego_yaw, roll=0.0)
        )
        lead_model = config.other_actors[1].model if len(config.other_actors) > 1 else "vehicle.nissan.patrol"
        self.lead_vehicle = CarlaDataProvider.request_new_actor(lead_model, lead_spawn_transform)
        
        if self.lead_vehicle:
            light_state = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
            self.lead_vehicle.set_light_state(carla.VehicleLightState(light_state))
            self.lead_vehicle.apply_control(carla.VehicleControl(hand_brake=False, brake=0.0))
            self.other_actors.append(self.lead_vehicle)
        
        ego = self.ego_vehicles[0]
        forward = ego.get_transform().get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(
            forward.x * self.ego_initial_speed,
            forward.y * self.ego_initial_speed,
            forward.z * self.ego_initial_speed
        ))

 
    def _create_behavior(self):
        root = py_trees.composites.Parallel(
            name="RootBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        # 前车行驶
        lead_drive_branch = py_trees.composites.Sequence("LeadVehicleDriveBranch")
        if self.lead_vehicle and self.ego_vehicle:
            lead_drive = LeadVehicleDrive(self.lead_vehicle, self.ego_vehicle, self.lead_vehicle_speed)
        else:
            lead_drive = py_trees.behaviours.Success(name="NoLeadVehicle")
        lead_drive_branch.add_child(lead_drive)

        # 场景逻辑
        scenario_logic_branch = py_trees.composites.Sequence("ScenarioLogicBranch")
        trigger_dist = InTriggerDistanceToVehicle(self.ego_vehicle, self.lead_vehicle, distance=self.trigger_distance_ego_lead)
        
        if self.lead_vehicle:
            continuous_lane_change = ContinuousLaneChange(
                self.lead_vehicle,
                target_lateral_distance=self.lane_change_total_distance,
                angle_max=self.lane_change_angle_max
            )
        else:
            continuous_lane_change = py_trees.behaviours.Success(name="NoLeadVehicle")
            
        end_condition = DriveDistance(self.ego_vehicle, distance=150)

        scenario_logic_branch.add_child(trigger_dist)
        scenario_logic_branch.add_child(continuous_lane_change)
        scenario_logic_branch.add_child(end_condition)

        root.add_child(lead_drive_branch)
        root.add_child(scenario_logic_branch)

        root.add_child(PassiveEgoSpeedHold(self.ego_vehicles[0], self.ego_initial_speed))
        return root

    # ===================== 以下是 100% 适配你场景的评估函数 =====================
    def _create_test_criteria(self):
        criteria = []
        if not self.ego_vehicle or not self.accident_vehicle:
            return criteria

        # 条件1：距离障碍车15米内必须减速
        criteria.append(StaticObstacleSlowDownCriterion(
            actor=self.ego_vehicle,
            hazard_actor=self.accident_vehicle,
            trigger_distance=15.0,
            decel_threshold=3.0,
            min_speed_after=5.0
        ))

        # 条件2：绝对不能碰撞
        criteria.append(StaticObstacleNoCollisionCriterion(
            actor=self.ego_vehicle,
            hazard_actor=self.accident_vehicle
        ))

        # 条件3：必须安全变道通过
        criteria.append(StaticObstacleSafePassCriterion(
            actor=self.ego_vehicle,
            hazard_actor=self.accident_vehicle,
            route_center_y=self.global_y
        ))

        return criteria


    def __del__(self):
        self.remove_all_actors()
