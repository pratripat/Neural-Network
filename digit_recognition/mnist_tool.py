import tkinter as tk
import numpy as np
from PIL import Image
from neural_network import Neural_Network
from pprint import pprint

class MNISTDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        self.canvas_size = 280  # Scaled for better visibility
        self.img_size = 28  # Target MNIST size
        self.brush_size = 10
        
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack()
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        
        self.save_button = tk.Button(self.button_frame, text="Save Image", command=self.save)
        self.save_button.pack(side=tk.LEFT)

        self.guess_button = tk.Button(self.button_frame, text="Guess Number", command=self.guess)
        self.guess_button.pack(side=tk.LEFT, expand=True, padx=10)
        
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT)

        self.entry_box = tk.Entry(root, width=30)
        self.entry_box.pack(pady=10)
        self.entry_box.bind("<Return>", self.get_input)
        
        self.image = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))  # Transparent background
        self.pixels = self.image.load()

        self.nn = Neural_Network([10,10], ['relu', 'softmax'])
        self.nn.load_weights('best_trained_weights.json')

    def paint(self, event):
        x1, y1 = event.x - self.brush_size, event.y - self.brush_size
        x2, y2 = event.x + self.brush_size, event.y + self.brush_size
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        
        for i in range(x1, x2):
            for j in range(y1, y2):
                if 0 <= i < self.canvas_size and 0 <= j < self.canvas_size:
                    self.pixels[i, j] = (0, 0, 0, 255)  # Black digit with full opacity
    
    def save(self):
        # img = self.image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        # img_array = np.array(img, dtype=np.uint8)
        # alpha_channel = img_array[:, :, 3]  # Extract alpha channel
        # black_image = np.full((self.img_size, self.img_size, 4), (0, 0, 0, 0), dtype=np.uint8)
        # black_image[:, :, 3] = alpha_channel  # Keep only alpha changes
        # final_img = Image.fromarray(black_image, mode='RGBA')

        final_img, alpha_channel = self.get_resized_image()
        final_img.save("mnist_digit.png")
        np.save("mnist_digit.npy", alpha_channel)
        print("Image saved as mnist_digit.npy and mnist_digit.png with black digit and transparency")
    
    def guess(self):
        final_img, alpha_channel = self.get_resized_image()

        pixels = alpha_channel.reshape((1,784))

        output = list(self.nn.forward(pixels)[0])
        dict = {index : float(output[index]) for index in range(len(output))}
        print(output.index(max(output)))
        pprint(dict, indent=4)
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 0))  # Reset to transparent
        self.pixels = self.image.load()

    def get_input(self, event=None):
        expected_number = int(self.entry_box.get())
        # will train the neural network to become better

    def get_resized_image(self):
        img = self.image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)
        alpha_channel = img_array[:, :, 3]  # Extract alpha channel
        black_image = np.full((self.img_size, self.img_size, 4), (0, 0, 0, 0), dtype=np.uint8)
        black_image[:, :, 3] = alpha_channel  # Keep only alpha changes
        final_img = Image.fromarray(black_image, mode='RGBA')

        return final_img, alpha_channel

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTDrawingApp(root)
    root.mainloop()
