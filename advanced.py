import cv2
import numpy as np

class advanced():
    
    @staticmethod
    def remove_green_screen(image):
      
        if image is None:
            print("Error: Unable to read the image.")
            return None
        
         
        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        # Define the green screen color range
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
    
        # Create a mask for the green screen color range
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
        # Invert the mask to keep the non-green areas
        inverted_mask = cv2.bitwise_not(mask)
    
        # Create a black background image
        black_background = np.zeros_like(image)
    
        # Replace the green screen area with the replacement color
        result = cv2.bitwise_and(image, image, mask=inverted_mask)
        result += cv2.bitwise_and(black_background, black_background, mask=mask)
  
        return result
    
    @staticmethod
    def invisibility_cloak(video_path,background_path):
        
        # cap = cv2.VideoCapture(video_path)
        cap = cv2.VideoCapture(1)
    
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return
    
        # Define the color range for the cloak (red color)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
    
        while True:
            ret, frame = cap.read()
    
            if not ret:
                break
    
            # Convert the frame from BGR to HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
            # Create a mask for the red color range
            mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    
            # Invert the mask to keep the non-red areas
            inverted_mask = cv2.bitwise_not(mask)
    
            # Replace the cloak color with the background (assuming a static background)
            background = cv2.imread(background_path)  # Replace with the actual path
            result = cv2.bitwise_and(frame, frame, mask=inverted_mask)
            result += cv2.bitwise_and(background, background, mask=mask)
    
            # Display the result
            cv2.imshow("Invisibility Cloak", result)
    
            if cv2.waitKey(30) & 0xFF == ord("q"):  # Press 'Esc' to exit
                break
    
        cap.release()
        cv2.destroyAllWindows()
    
if __name__=="__main__":
    
    #test invisibilty
    # advanced.invisibility_cloak("background.mp4","background.jpg")
   
    
    #test the green screen
    image = cv2.imread("green_screen_test.jpg")
    
    result = advanced.remove_green_screen(image)
    
    # Show the result
    cv2.imshow("Original Image", image)
    cv2.imshow("Green Screen Removal", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # cap = cv2.VideoCapture(1)
    
    # if not cap.isOpened():
    #     print("Error: Unable to open video.")
    #     exit(0)
        
    # while True:
    #         ret, frame = cap.read()
    
    #         if not ret:
    #             break    
            
    #         result = advanced.remove_green_screen(frame)
            
    #         cv2.imshow("result",result)
            
            
    #         if cv2.waitKey(30) & 0xFF == ord("q"):  # Press 'Esc' to exit
    #             break
    
    # cap.release()
    # cv2.destroyAllWindows()