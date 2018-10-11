import numpy as np

from Box2D import (
    b2CircleShape,
    b2EdgeShape,
    b2PolygonShape,
    b2_staticBody,
    b2_kinematicBody,
    b2DistanceJoint,
    b2PulleyJoint,
    b2MouseJoint,
    b2RevoluteJoint,
    b2PrismaticJoint,
    b2Transform,
    b2Vec2,
    b2Rot,
    b2WeldJoint,
)
from pyglet.window import key

from openlock import rendering
from openlock.common import Color, TwoDConfig, COLORS
from openlock.kine import TwoDKinematicTransform
from openlock.settings_render import RENDER_SETTINGS

VIEWPORT_W = 800
VIEWPORT_H = 800
SCALE = 25.0  # affects how fast-paced the game is, forces should be adjusted as well


def screen_to_world_coord(xy):
    x_world = (xy[0] - VIEWPORT_W / 2) / (SCALE / 2.0)
    y_world = (xy[1] - VIEWPORT_H / 2) / (SCALE / 2.0)
    return (x_world, y_world)


class Box2DRenderer:
    def __init__(self, enter_key_callback):
        self.viewer = rendering.Viewer(
            VIEWPORT_W, VIEWPORT_H, pre_render_callbacks=[self._draw_last_arrow]
        )
        self.viewer.set_bounds(
            -VIEWPORT_W / SCALE,
            VIEWPORT_W / SCALE,
            -VIEWPORT_H / SCALE,
            VIEWPORT_H / SCALE,
        )
        self.viewer.window.push_handlers(
            self.on_mouse_drag,
            self.on_mouse_press,
            self.on_mouse_release,
            self.on_key_press,
        )

        self.enter_key_callback = enter_key_callback

        self.reset()

        # TODO: registry decorator
        self.on_mouse_press_callbacks, self.on_mouse_release_callbacks = (
            {self._detect_region_click},
            {},
        )

    def register_clickable_region(self, clickable_region):
        self.clickable_regions.add(clickable_region)

    def close(self):
        self.viewer.close()

    # event callbacks
    def on_mouse_press(self, x, y, button, modifiers):
        for callback in self.on_mouse_press_callbacks:
            callback(x, y, button, modifiers)

    def on_mouse_release(self, x, y, button, modifiers):
        for callback in self.on_mouse_release_callbacks:
            callback(x, y, button, modifiers)

    def _detect_region_click(self, x, y, button, modifiers):
        for clickable_region in self.clickable_regions:
            if clickable_region.test_region(screen_to_world_coord((x, y))):
                clickable_region.call()

    def _set_config_on_mouse_press(self, x, y, button, modifiers):
        self.arrow_start = (x, y)

    def _set_config_on_mouse_release(self, x, y, button, modifiers):
        self.arrow_end = (x, y)
        # compute arrow
        theta = np.arctan2(
            self.arrow_end[1] - self.arrow_start[1],
            self.arrow_end[0] - self.arrow_start[0],
        )
        screen_arrow_start = screen_to_world_coord(self.arrow_start)
        self.desired_config = TwoDConfig(
            screen_arrow_start[0], screen_arrow_start[1], theta
        )

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.cur_arrow_end = (x, y)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ENTER or symbol == key.RETURN:
            self.enter_key_callback()

    def _draw_last_arrow(self):
        if self.arrow_start and self.cur_arrow_end:
            self.viewer.draw_line(
                screen_to_world_coord(self.arrow_start),
                screen_to_world_coord(self.cur_arrow_end),
            )
        self.viewer.draw_line((50, 50), (53, 53))

    def reset(self):
        self.markers = dict()
        self.clickable_regions = set()
        self.cur_arrow_end = (
            self.arrow_start
        ) = self.arrow_end = self.desired_config = None

    def _draw_arrow(self, *args):
        x, y, theta, width, length, color = args

        rot = b2Rot(theta)
        trans = b2Transform((x, y), rot)

        vert = [b2Vec2((0, -width / 2)), b2Vec2((0, width / 2)), b2Vec2((length, 0))]
        trans_vert = [trans * v for v in vert]

        self.viewer.draw_polygon(trans_vert, filled=True, color=color)
        self.viewer.draw_polygon(trans_vert, filled=False)

    def _draw_cross(self, *args):
        x, y, size, color = args

        self.viewer.draw_line((x - size, y - size), (x + size, y + size), color=color)
        self.viewer.draw_line((x - size, y + size), (x + size, y - size), color=color)

    def render_multiple_worlds(self, worlds, mode="human"):
        for world in worlds:
            self._render_world(world, mode)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _render_world(self, world, mode):
        # for joint in world.joints:
        #    if type(joint.userData) is dict and 'obj_type' in joint.userData and joint.userData['obj_type'] == 'lock_joint':
        #        bodyA, bodyB = joint.bodyA, joint.bodyB
        #        xf1, xf2 = bodyA.transform, bodyB.transform
        #        x1, x2 = xf1.position, xf2.position
        #        p1, p2 = joint.anchorA, joint.anchorB
        #        padding = joint.userData['plot_padding']
        #        width = 0.5

        #        # plot the bounds in which body A's anchor point can move relative to B
        #        axis = joint.userData['joint_axis']
        #        local_axis = joint.bodyA.GetLocalVector(axis)
        #        world_axis = joint.bodyA.GetWorldVector(local_axis)
        #        lower_lim, upper_lim = joint.limits
        #        end1 = p2 - world_axis * (upper_lim + padding)
        #        end2 = p2 - world_axis * (lower_lim - padding)
        #        norm = b2Vec2(-world_axis[1], world_axis[0])

        #        vertices = [end1 + norm * width, end1 - norm * width, end2 - norm * width, end2 + norm * width]
        #        self.viewer.draw_polygon(vertices, filled=True, color=RENDER_SETTINGS['COLORS']['pris_joint'])
        # draw bodies
        if RENDER_SETTINGS["DRAW_SHAPES"]:
            for body in world.bodies:
                transform = body.transform
                for fixture in body.fixtures:
                    shape = fixture.shape

                    # if the userData used a color, use that color
                    if body.userData is not None and body.userData.color is not None:
                        color = body.userData.color
                    elif not body.active:
                        color = RENDER_SETTINGS["COLORS"]["active"]
                    elif body.type == b2_staticBody:
                        color = RENDER_SETTINGS["COLORS"]["static"]
                    elif body.type == b2_kinematicBody:
                        color = RENDER_SETTINGS["COLORS"]["kinematic"]
                    elif not body.awake:
                        color = RENDER_SETTINGS["COLORS"]["asleep"]
                    else:
                        color = RENDER_SETTINGS["COLORS"]["default"]

                    if isinstance(fixture.shape, b2EdgeShape):
                        self.viewer.draw_line(
                            fixture.shape.vertices[0],
                            fixture.shape.vertices[1],
                            color=color,
                        )
                    elif isinstance(fixture.shape, b2CircleShape):
                        # print fixture.body.transform
                        trans = rendering.Transform(
                            translation=transform * fixture.shape.pos
                        )
                        self.viewer.draw_circle(
                            fixture.shape.radius, filled=True, color=color
                        ).add_attr(trans)
                        self.viewer.draw_circle(
                            fixture.shape.radius, filled=False
                        ).add_attr(trans)
                    elif isinstance(fixture.shape, b2PolygonShape):
                        vertices = [transform * v for v in fixture.shape.vertices]
                        self.viewer.draw_polygon(vertices, filled=True, color=color)
                        self.viewer.draw_polygon(vertices, filled=False)

        # draw joints
        if RENDER_SETTINGS["DRAW_JOINTS"]:
            for joint in world.joints:
                self.__draw_joint(joint)

        # draw markers
        if RENDER_SETTINGS["DRAW_MARKERS"]:
            for _, val in list(self.markers.items()):
                type_, args = val
                if type_ == "arrow":
                    self._draw_arrow(*args)
                # elif type == 'cross':
                #     self._draw_cross(args)

    def __draw_joint(self, joint, color=Color(0.5, 0.8, 0.8)):
        """
        Draw any type of joint
        """
        bodyA, bodyB = joint.bodyA, joint.bodyB
        xf1, xf2 = bodyA.transform, bodyB.transform
        x1, x2 = xf1.position, xf2.position
        p1, p2 = joint.anchorA, joint.anchorB

        if isinstance(joint, b2DistanceJoint):
            self.viewer.draw_line(p1, p2, color=RENDER_SETTINGS["COLORS"]["dist_joint"])
        # elif isinstance(joint, b2PulleyJoint):
        #     s1, s2 = joint.groundAnchorA, joint.groundAnchorB
        #     self.viewer.draw_line(s1, p1, color=RENDER_SETTINGS['COLORS'][''])
        #     self.viewer.draw_line(s2, p2, color=RENDER_SETTINGS['COLORS'][''])
        #     self.viewer.draw_line(s1, s2, color=RENDER_SETTINGS['COLORS'][''])
        elif isinstance(joint, b2RevoluteJoint):
            trans = rendering.Transform(translation=p1)
            self.viewer.draw_circle(
                0.5, fillied=True, color=RENDER_SETTINGS["COLORS"]["rev_joint"]
            ).add_attr(trans)
        elif isinstance(joint, b2WeldJoint):
            trans = rendering.Transform(translation=p2)
            self.viewer.draw_circle(
                0.5, fillied=True, color=RENDER_SETTINGS["COLORS"]["weld_joint"]
            ).add_attr(trans)

        else:
            pass
            # self.viewer.draw_line(x1, p1, color=color)
            # self.viewer.draw_line(p1, p2, color=color)
            # self.viewer.draw_line(x2, p2, color=color)
