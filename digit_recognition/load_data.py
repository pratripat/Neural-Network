import threading, queue, os, random
from PIL import Image
import numpy as np

def load_thread_inputs(folders, task_id, result_queue, number_of_threads=2):
    print(f'Thread {task_id} starting...')
    inputs = []
    targets = []

    mult = 5
    total_inputs = 10773 * len(folders) * mult
    # # total_inputs = 50000
    # max_counter = 10772*mult

    # def load_images(internal_thread_id, folder_path, images, mult, loaded_images_queue):
    #     inputs = []
    #     targets = []

    #     print(f'\tInternal thread {internal_thread_id}: set up done...')

    #     for image_file in os.listdir(folder_path):
    #         image = Image.open(folder_path+'/'+image_file)
    #         image_array = np.array(image)

    #         modified_image_arrays = []

    #         for _ in range(mult):
    #             # Translate the image
    #             width, height = image.size
    #             new_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # Black background
    #             # offset = (random.randint(-50, 50), random.randint(-50,50))
    #             offset = (0,0)
    #             new_image.paste(image, offset)

    #             pixels = new_image.load()

    #             for _ in range(random.randint(10, 20)):
    #                 x = random.randrange(0, width)
    #                 y = random.randrange(0, height)

    #                 pixels[x, y] = (0, 0, 0, int(random.uniform(0,1)*255))

    #             modified_image_arrays.append(np.array(new_image))

    #         for image_array in modified_image_arrays:
    #             pixels = []

    #             for row in image_array:
    #                 for pixel in row:
    #                     pixels.append(pixel[3])

    #             inputs.append(pixels)
    #             targets.append(int(folder))

    #     loaded_images_queue.put((inputs, targets))
    #     print(f'\tInternal Thread {internal_thread_id}: done loaded images!')

    path = 'dataset'
    for folder in os.listdir(path):
        if folder not in folders: continue

        # counter = 0

        folder_path = path+'/'+folder+'/'+folder
        # threads = []
        # image_files = os.listdir(folder_path)
        # length = len(image_files) // number_of_threads

        # loaded_images_queue = queue.Queue()

        # for i in range(number_of_threads):

        #     # makes sure that all images get loaded if the number of images is not divisble by number of threads 
        #     end = (i+1)*length
        #     if i+1 == number_of_threads:
        #         end = len(image_files)

        #     # setting up the thread 
        #     thread = threading.Thread(target=load_images, args=(f'{task_id}->{i+1}', folder_path, image_files[i*length : end], mult, loaded_images_queue))
        #     threads.append(thread)
        #     thread.start()
        
        # for thread in threads:
        #     thread.join()
        
        # while not loaded_images_queue.empty():
        #     (thread_inputs, thread_targets) = loaded_images_queue.get()
        #     inputs.extend(thread_inputs)
        #     targets.extend(thread_targets)


        for image_file in os.listdir(folder_path):
            image = Image.open(folder_path+'/'+image_file)
            image_array = np.array(image)

            modified_image_arrays = []

            for _ in range(mult):
                # Translate the image
                width, height = image.size
                new_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # Black background
                # offset = (random.randint(-50, 50), random.randint(-50,50))
                offset = (0,0)
                new_image.paste(image, offset)

                pixels = new_image.load()

                for _ in range(random.randint(10, 20)):
                    x = random.randrange(0, width)
                    y = random.randrange(0, height)

                    pixels[x, y] = (0, 0, 0, int(random.uniform(0,1)*255))

                modified_image_arrays.append(np.array(new_image))

            for image_array in modified_image_arrays:
                pixels = []

                for row in image_array:
                    for pixel in row:
                        pixels.append(pixel[3])

                inputs.append(pixels)
                targets.append(int(folder))

                # counter += 1
                # if counter == max_counter:
                #     counter = 0
                #     break

    result_queue.put((task_id, (inputs, targets, total_inputs)))
    print(f'Thread {task_id}: Loading files {folders} done successfully!')

    # return inputs, targets, total_inputs

def shuffle_data(inputs, targets):
    # randomizing the order of inputs, targets
    temp = list(zip(inputs, targets))
    random.shuffle(temp)

    inputs, targets = zip(*temp)
    inputs = np.array(list(inputs))
    targets = np.array(list(targets))

    return inputs, targets

def load_mnist_dataset():
    result_queue = queue.Queue()

    threads = []
    # for i in range(5):
    #     thread = threading.Thread(target=load_thread_inputs, args=([str(i*2), str(i*2 + 1)], i, result_queue))
    #     threads.append(thread)
    #     thread.start()

    for i in range(10):
        thread = threading.Thread(target=load_thread_inputs, args=([str(i)], i+1, result_queue))
        threads.append(thread)
        thread.start()


    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    inputs = []
    targets = []
    total_inputs = 0
    while not result_queue.empty():
        task_id, (thread_inputs, thread_targets, thread_total_inputs) = result_queue.get()
        inputs.extend(thread_inputs)
        targets.extend(thread_targets)
        total_inputs += thread_total_inputs

    inputs, targets = shuffle_data(inputs, targets)

    return inputs, targets, total_inputs

def convert_to_image(pixels):
    image_array = np.array([[[0, 0, 0, pixels[i*28 + j]] for j in range(28)] for i in range(28)], dtype=np.uint8)
    image = Image.fromarray(image_array)

    return image
