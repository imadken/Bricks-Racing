# median and mean
# gauss laplacien
# erosion et dilatation aka morphologiques
import copy
import numpy as np
import cv2
from Metrics import Metrics
from convolution import Convolution
from time import time

class Filters():

        
    @staticmethod
    def filtreMoy(img, vois=7):
        
        h, w = img.shape
        imgtoy = np.zeros(img.shape, img.dtype)
        
        y=0
        
        while(y<h):
            x=0
            while(x<w):
                if(y < (vois/2) or y > (h - vois/2) or x < (vois/2) or x > (w - vois/2)):
                    imgtoy[y, x] = img[y, x]
                else:
                    n = int(vois/2)
                    imgvois = img[int(y - n):int(y + n + 1), int(x - n):int(x+n+1)]
                    # imgtoy[y, x] = np.mean(imgvois)
                    imgtoy[y, x] = Metrics.mean(imgvois)
                x+=1
            y+=1
            # if (y%10 == 0):
                # print(y)
                        
        return imgtoy
    
    
    @staticmethod
    def filtreMed(img, vois=7):
        
        h, w = img.shape
        imgmed = np.zeros(img.shape, img.dtype)
        
        y=0
        
        while(y<h):
            x=0
            while(x<w):
                if(y < (vois/2) or y > (h - vois/2) or x < (vois/2) or x > (w - vois/2)):
                    imgmed[y, x] = img[y, x]
                else:
                    n = int(vois/2)
                    imgvois = img[int(y - n):int(y + n + 1), int(x - n):int(x+n+1)]
                    
                    #get voisinage
                    t = np.zeros((vois*vois), img.dtype)
                    hv,wv=imgvois.shape
                    
                    yv=0
                    while yv < hv:
                        xv=0
                        while xv < wv:
                            # t[yv*vois + xv] = imgmed[yv, xv]
                            t[yv*vois + xv] = imgvois[yv, xv]
                            xv += 1
                        yv += 1
                        
                    imgmed[y,x]= Metrics.median(t) 
                      
                x+=1
            y+=1
            # if (y%10 == 0):
                # print(y)
                        
        return imgmed
    
    @staticmethod
    def filter2D(img, kernel):
         
         """an implementation of CV2.filter2D for linear filters
 
         Returns:
             result img: numpy array of the filtred input image 
         """
         img_height, img_width = img.shape
         kernel_height, kernel_width = kernel.shape
     
         # Ensure the kernel has an odd size
         assert kernel_height % 2 == 1 and kernel_width % 2 == 1, "Kernel size must be odd"
     
         # Calculate the padding needed for convolution
         pad_y = kernel_height // 2
         pad_x = kernel_width // 2
     
         # Create an empty output image
         img_result = np.zeros_like(img)
         
         # Iterate over each pixel in the input image
         y=pad_y
         while y < img_height - pad_y:
             x=pad_x
             while x < img_width - pad_x :
                 # Extract the local region around the pixel
                 img_region = img[y - pad_y:y + pad_y + 1, x - pad_x:x + pad_x + 1]
                 
                 # Perform the convolution
                 img_result[y, x] = Convolution.convolution(img_region,kernel)
                 x += 1
                 
             y += 1 
                 
         return img_result
     
    @staticmethod
    def filter_Morph(img, kernel,type="EROSION",iterations=1):
         """Performs either dilatation or erosion
 
         Args:
             img (ndarray): image tob eused
             kernel (ndarray): cross or rect
             type (str, optional): Erosion or Dilate. Defaults to "EROSION".
 
         Returns:
             ndarray: image after Filter
         """
         if type.lower()=="erosion":
             method = Convolution.erode
         else:
            method = Convolution.dilate
            
         
         img_height, img_width = img.shape
         kernel_height, kernel_width = kernel.shape
     
         # Ensure the kernel has an odd size
         assert kernel_height % 2 == 1 and kernel_width % 2 == 1, "Kernel size must be odd"
     
         # Calculate the padding needed for convolution
         pad_y = kernel_height // 2
         pad_x = kernel_width // 2
         img_result = img.copy()
        #  img_result = np.zeros_like(img)
         # Iterate over each pixel in the input image
         while iterations > 0:
            y=pad_y
            while y < img_height - pad_y:
                x=pad_x
                while x < img_width - pad_x :
                    # Extract the local region around the pixel
                    img_region = img[y - pad_y:y + pad_y + 1, x - pad_x:x + pad_x + 1]
                    
                    # Perform the convolution
                    img_result[y, x] = method(img_region,kernel)
                    
                    x += 1  
                y += 1
                if y%10==0 :print(y)
            
            iterations -= 1 
                
         return img_result
     
    @staticmethod
    def Sobel(img, Hkernel,Vkernel):
         
         """an implementation of CV2.filter2D for linear filters
 
         Returns:
             result img: numpy array of the filtred input image 
         """
         
         img_copy = img.copy()
         img_copy=np.array(img_copy,dtype=np.float64)
         
         gradient_x = Filters.filter2D(img_copy,Hkernel)
         gradient_y = Filters.filter2D(img_copy,Vkernel)
         
        #  gradient_x = Filters.filter2D(img,Hkernel)
        #  gradient_y = Filters.filter2D(img,Vkernel)
        #  gradient_x = cv2.filter2D(img, cv2.CV_64F, Hkernel)
        #  gradient_y = cv2.filter2D(img, cv2.CV_64F, Vkernel)
         
         magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        #  magnitude = np.uint8(magnitude)
         
         
        #  magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
         
                 
         return magnitude 
   
if __name__=="__main__":
    img = cv2.imread("barca.png", cv2.IMREAD_GRAYSCALE)
    
    
    HSober_kernel = np.array([[-1, 0, 1],[ -2, 0, 2], [-1, 0, 1]])
    VSober_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) 
    emboss_kernel = np.array([[-2, -1, 0],[-1,  1, 1],[ 0,  1, 2]])
    laplacien_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    gauss_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    erosion_kernel = Convolution.create_structuring_element(shape="rect",size=25)
    
    # s=time()
    # img_res = Filters.filtreMoy(img)
    # img_res = Filters.filtreMed(img)
    # img_laplacien = Filters.filter2D(img,laplacien_kernel)
    img_emboss = Filters.filter2D(img,emboss_kernel)
    # img_gauss = Filters.filter2D(img,gauss_kernel)
    # img_binary = Convolution.threshold(img,128,255)
    # img_erode = Filters.filter_Morph(img_binary,erosion_kernel,type="erosion")
    # img_sobel = Filters.Sobel(img,HSober_kernel,VSober_kernel)
    # e =time()
    # print(e-s)
    # cv2.imshow("mean",img_res)
    # cv2.imshow("median",img_res)
    # cv2.imshow("laplacien",img_laplacien)
    # cv2.imshow("laplacien",img_gauss)
    # cv2.imshow("erode",img_erode)
    # cv2.imshow("binary",img_binary)
    cv2.imshow("before",img)
    # cv2.imshow("sobel",img_sobel)
    cv2.imshow("sobel",img_emboss)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()