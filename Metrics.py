from Sort import Sort

class Metrics():
    @staticmethod
    def mean(array):
        h,w = array.shape
        total,count=0,0
        
        y=0
        while(y<h):
            x=0
            while(x<w):
                total += array[y,x]
                count += 1
                x += 1
            y += 1    
        return (total/count)
    
    @staticmethod
    def median(array):
        # array.sort()
        
        array=Sort.quick_sort(array)
       
        if len(array)%2:
            return array[int(len(array)/2)]
        
        return int((array[int(len(array)/2)-1]+array[int(len(array)/2)])/2)
    
    
        

if __name__=="__main__":
    
    print(Metrics.median([134,50,32,626,32,10]))