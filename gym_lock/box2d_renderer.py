from gym_lock import rendering
from gym_lock.common import Color, TwoDConfig
from Box2D import b2CircleShape, b2EdgeShape, b2PolygonShape, b2_staticBody, b2_kinematicBody, b2DistanceJoint, \
    b2PulleyJoint, b2MouseJoint, b2RevoluteJoint, b2PrismaticJoint
import numpy as np
import Box2D as b2
from pyglet.window import key
COLORS = {
    'active': Color(0.5, 0.5, 0.3),
    'static': Color(0.5, 0.9, 0.5),
    'kinematic': Color(0.5, 0.5, 0.9),
    'asleep': Color(0.6, 0.6, 0.6),
    'default': Color(0.9, 0.7, 0.7),
}

VIEWPORT_W = 800
VIEWPORT_H = 800
SCALE = 25.0  # affects how fast-paced the game is, forces should be adjusted as well

def screen_to_world_coord(xy):
    x_world = (xy[0] - VIEWPORT_W / 2) / (SCALE / 2.0)
    y_world = (xy[1] - VIEWPORT_H / 2) / (SCALE / 2.0)
    return (x_world, y_world)

class Box2DRenderer():

    def __init__(self, enter_key_callback):
        self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H, pre_render_callbacks=[self.__draw_last_arrow])
        self.viewer.set_bounds(-VIEWPORT_W / SCALE, VIEWPORT_W / SCALE, -VIEWPORT_H / SCALE, VIEWPORT_H / SCALE)
        self.viewer.window.push_handlers(self.on_mouse_drag,
                                         self.on_mouse_press,
                                         self.on_mouse_release,
                                         self.on_key_press)

        self.enter_key_callback = enter_key_callback

        self.cur_arrow_end = self.arrow_start = self.arrow_end = self.desired_config = None

    def close(self):
        self.viewer.close()

    # event callbacks
    def on_mouse_press(self, x, y, button, modifiers):
        self.arrow_start = (x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        self.arrow_end = (x, y)
        # compute arrow
        theta = np.arctan2(self.arrow_end[1] - self.arrow_start[1], self.arrow_end[0] - self.arrow_start[0])
        screen_arrow_start = screen_to_world_coord(self.arrow_start)
        self.desired_config = TwoDConfig(screen_arrow_start[0],
                                         screen_arrow_start[1],
                                         theta)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.cur_arrow_end = (x, y)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ENTER or symbol == key.RETURN:
            self.enter_key_callback()

    def __draw_last_arrow(self):
        if self.arrow_start and self.cur_arrow_end:
            self.viewer.draw_line(screen_to_world_coord(self.arrow_start), screen_to_world_coord(self.cur_arrow_end))
        self.viewer.draw_line((50, 50), (53, 53))


    def render_world(self, world, mode='human'):

        # draw bodies
        for body in world.bodies:
            transform = body.transform
            for fixture in body.fixtures:
                shape = fixture.shape

                if not body.active:
                    color = COLORS['active']
                elif body.type == b2_staticBody:
                    color = COLORS['static']
                elif body.type == b2_kinematicBody:
                    color = COLORS['kinematic']
                elif not body.awake:
                    color = COLORS['asleep']
                else:
                    color = COLORS['default']

                if isinstance(fixture.shape, b2EdgeShape):
                    self.viewer.draw_line(fixture.shape.vertices[0], fixture.shape.vertices[1], color=color)
                elif isinstance(fixture.shape, b2CircleShape):
                    # print fixture.body.transform
                    trans = rendering.Transform(translation=transform * fixture.shape.pos)
                    self.viewer.draw_circle(fixture.shape.radius, filled=True, color=color).add_attr(trans)
                    self.viewer.draw_circle(fixture.shape.radius, filled=False).add_attr(trans)
                elif isinstance(fixture.shape, b2PolygonShape):
                    vertices = [transform * v for v in fixture.shape.vertices]
                    self.viewer.draw_polygon(vertices, filled=True, color=color)
                    self.viewer.draw_polygon(vertices, filled=False)

        # draw joints
        for joint in world.joints:
            self.__draw_joint(joint)

        return self.viewer.render(return_rgb_array= mode == 'rgb_array')

    def __draw_joint(self, joint):
        """
        Draw any type of joint
        """
        bodyA, bodyB = joint.bodyA, joint.bodyB
        xf1, xf2 = bodyA.transform, bodyB.transform
        x1, x2 = xf1.position, xf2.position
        p1, p2 = joint.anchorA, joint.anchorB
        color = Color(0.5, 0.8, 0.8)

        if isinstance(joint, b2DistanceJoint):
            self.viewer.draw_line(p1, p2, color=color)
        elif isinstance(joint, b2PulleyJoint):
            s1, s2 = joint.groundAnchorA, joint.groundAnchorB
            self.viewer.draw_line(s1, p1, color=color)
            self.viewer.draw_line(s2, p2, color=color)
            self.viewer.draw_line(s1, s2, color=color)
        elif isinstance(joint, b2MouseJoint):
            pass  # don't draw it here
        elif isinstance(joint, b2RevoluteJoint):
            trans = rendering.Transform(translation=p1)
            self.viewer.draw_circle(0.5, fillied=True).add_attr(trans)
        elif isinstance(joint, b2PrismaticJoint):
            # TODO: implement this
            pass
        else:
            self.viewer.draw_line(x1, p1, color=color)
            self.viewer.draw_line(p1, p2, color=color)
            self.viewer.draw_line(x2, p2, color=color)