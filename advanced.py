import cv2
import numpy as np
from Color import Color


class advanced():

    @staticmethod
    def remove_green_screen(image,lower = np.array([40, 40, 40]),upper = np.array([80, 255, 255])):
      
        if image is None:
            print("Error: Unable to read the image.")
            return None

        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask for the green screen color range,boundaries must be in hsv
        # mask = cv2.inRange(hsv_image, lower, upper)
        mask = Color.my_in_range(hsv_image, lower, upper)
        
    
        # Invert the mask to keep the non-green areas
        inverted_mask = cv2.bitwise_not(mask)
    
        # Create a black background image
        black_background = np.zeros_like(image)
    
        # Replace the green screen area with the replacement color
        result = cv2.bitwise_and(image, image, mask=inverted_mask)
        result += cv2.bitwise_and(black_background, black_background, mask=mask)
        
        # cv2.circle(result,(300,250),20,(0,255,0),5)
        # print(hsv_image[300,250])
        return result
   
    
    @staticmethod
    # def invisibility_cloak_frame(frame,background_path,lower = np.array([0,  80, 200]),upper = np.array([180, 100 ,230])):
    # def invisibility_cloak_frame(frame,background_path="background/background.jpg",lower = np.array([95,30,130]),upper = np.array([120,120,170])): grey
    # def invisibility_cloak_frame(frame,background_path="background/background.jpg",lower = np.array([150,95,215]),upper = np.array([180,130,260])):
    def invisibility_cloak_frame(frame,background_path="background/background.jpg",lower = np.array([120,10,160]),upper = np.array([170,120,245])):
      
        # Convert the frame from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
        # Create a mask for the red color range
        # mask = cv2.inRange(hsv_frame, lower, upper) #OPENCV
        mask = Color.my_in_range(hsv_frame, lower, upper)
      
        # cv2.imshow("mask",mask)
        
        # Invert the mask to keep the non-red areas
        inverted_mask = cv2.bitwise_not(mask)

        # Replace the cloak color with the background (assuming a static background)
        background = cv2.imread(background_path)  # Replace with the actual path
        result = cv2.bitwise_and(frame, frame, mask=inverted_mask)
        result += cv2.bitwise_and(background, background, mask=mask)
        
        #Uncomment to set the color
        # cv2.circle(result,(300,250),20,(0,255,0),5)
        # print(hsv_frame[300,250])
        
        return result
    
    @staticmethod
    def main(choice="Green screen"):
        
        if choice.lower == "Green screen":
            method = advanced.remove_green_screen
        else:
            method = advanced.invisibility_cloak_frame    
        
        # cap = cv2.VideoCapture(video_path)
        cap = cv2.VideoCapture(1)
    
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return
        
        while True:
            ret, frame = cap.read()
    
            if not ret:
                break
    
            result = method(frame)
            # Display the result
            cv2.imshow(choice, result)
    
            if cv2.waitKey(30) & 0xFF == ord("q"):  # Press 'Esc' to exit
                break
    
        cap.release()
        cv2.destroyAllWindows()
           
    
if __name__=="__main__":
    
  
    advanced.main(choice="invisibilty")
   
    