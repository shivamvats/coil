from hydra.utils import to_absolute_path
from minigrid.core.world_object import *
from PIL import Image


class ADIKey(Key):
    """
    Adds an additional feature of 'scale' to minigrid Key class
    """
    def __init__(self, color: str = "blue", scale=1.0):
        self.scale = scale
        super().__init__( color)

    def render(self, img):
        c = COLORS[self.color]
        s = self.scale

        # Vertical quad
        fill_coords(img, point_in_rect(s*0.50, s*0.63, s*0.31, s*0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(s*0.38, s*0.50, s*0.59, s*0.66), c)
        fill_coords(img, point_in_rect(s*0.38, s*0.50, s*0.81, s*0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=s*0.56, cy=s*0.28, r=s*0.190), c)
        fill_coords(img, point_in_circle(cx=s*0.56, cy=s*0.28, r=s*0.064), (0, 0, 0))


class Human(WorldObj):
    """Visualizes a human object in the grid.

    XXX I think the right approach is to modify the vis code to visualize human
    as an agent and not treat him as an object..
    """
    def __init__(self, color: str = "blue"):
        super().__init__("key", color)

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        # load image
        image_path = to_absolute_path("data/gridworld/human_basic.jpeg")
        human_image = Image.open(image_path)

        # Convert the image to a numpy array
        human_image = np.array(human_image)

        # Assuming img is a numpy array, you can overlay the loaded image
        # Here we just copy the loaded image to the top-left corner of img
        img[:human_image.shape[0], :human_image.shape[1]] = human_image
