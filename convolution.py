import cv2
import numpy as np
class Convolution:
    @staticmethod
    def convolution(img,kernel):
        
        h,w=img.shape
        sum_value = 0
        
        y=0
        while y<h:
            x=0
            while x<w:
                sum_value += img[y,x] * kernel[y,x]
                x+=1
            y+=1     
        return sum_value 
     
    @staticmethod
    def threshold(img, thresh, max_val):
         """
         Apply binary thresholding to an image.
     
         Parameters:
             - img: Input image (grayscale).
             - thresh: Threshold value.
             - max_val: Maximum value to use with binary thresholding.
             - output: Output image to store the result.
     
         Returns:
             None (output is modified in place).
         """
         output=np.zeros_like(img)
         
         h, w = img.shape
         y=0    
         while y < h:
             x=0
             while x < w:
                     output[y,x] = max_val if img[y,x] > thresh else 0  
                     x += 1
             y += 1        
         return output 
     
    @staticmethod 
    def create_structuring_element(shape ="CROSS",size=1):
       # Ensure size is an odd number
       size = (size * 2) + 1
       # Create an empty matrix
       
       
       if shape.lower() == "cross" :
               kernel = np.zeros((size, size), dtype=np.uint8)
               kernel[:, size // 2] = 1
               kernel[size // 2, :] = 1
               
       else:
               kernel = np.ones((size, size), dtype=np.uint8)
               
       return kernel     
          
    @staticmethod
    def erode(img,kernel):
        h,w=img.shape
        kernel_height, kernel_width = kernel.shape
        sum_value = 1
        y=0
        while y<h:
            x=0
            while x<w:
                # if y != w // 2 and x != h // 2:
                if y != kernel_height // 2 and x != kernel_width // 2:
                   sum_value = img[y,x] & kernel[y,x] & sum_value
                x+=1
            y+=1     
        return sum_value 
    
    @staticmethod
    def dilate(img,kernel):
        h,w=img.shape
        kernel_height, kernel_width = kernel.shape
        sum_value = 0
        y=0
        while y<h:
            x=0
            while x<w:
                if y != kernel_height // 2 and x != kernel_width // 2:
                   sum_value = img[y,x] | kernel[y,x] | sum_value
                x+=1
            y+=1     
        return sum_value                   
  
if __name__=="__main__":
    img = cv2.imread("palestine.png", cv2.IMREAD_GRAYSCALE)
    
    img_res = Convolution.threshold(img,128,255)
  
    cv2.imshow("before",img)
    cv2.imshow("after",img_res)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()                     