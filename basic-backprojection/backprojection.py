from PIL import Image 
import numpy as np

print(1)
size=256

def display(img,data):
    global size
    image=np.reshape(data,(size*size))
    img.putdata(image) 
    img.show() 



def backproject_and_reconstruct(data):
    global size
    #forward projections
    projection_90=np.sum(data,0)/size
    projection_0=np.sum(data,1)/size

    print(projection_0,projection_90)

    #back projections
    back_90=np.tile(projection_90,(size,1))
    back_0=np.tile(np.reshape(projection_0,(size,1)),(1,size))

    print(back_0,back_90)

    #reconstructing

    data_re=(np.add(back_0,back_90)/2).astype(int)
    return data_re


def create_image_1(): ############################# half rectangle white,other half black
    ones=np.ones((size,int(size/2)))*256
    zeroes=np.zeros((size,int(size/2)))
    data=np.concatenate((ones,zeroes),1)
    print(data)
    return data

def create_image_2():  ############### black square in white background
    ones=np.ones((int(size/4),size))*256
    ones_s=np.ones((int(size/2),int(size/4)))*256
    zeroes=np.zeros((int(size/2),int(size/2)))
    mid=np.concatenate((ones_s,zeroes,ones_s),1)
    data=np.concatenate((ones,mid,ones))
    print(data)
    return data

def create_image_3(): ############ white square in black background
    zeros=np.zeros((int(size/4),size))
    zeros_s=np.zeros((int(size/2),int(size/4)))
    ones=np.ones((int(size/2),int(size/2)))*256
    mid=np.concatenate((zeros_s,ones,zeros_s),1)
    data=np.concatenate((zeros,mid,zeros))
    print(data)
    return data

def create_image_4():
    zeros=np.zeros((int(size/4),size))
    zeros_s=np.zeros((int(size/2),int(size/4)))
    ones=np.ones((int(size/2),int(size/2)))*256
    mid=np.concatenate((zeros_s,ones,zeros_s),1)
    data=np.concatenate((zeros,mid,zeros))
    img = Image.new("L", (size, size)) 
    img.putdata(np.reshape(data,(size*size)))
    img=img.rotate(45)
    data=np.array(img)
    print(data)
    return data


img = Image.new("L", (size, size)) 
data=create_image_4()
display(img,data)
img.save('orginal.jpg')

data_re=backproject_and_reconstruct(data)
print(data_re)
img = Image.new("L", (size, size)) 
display(img,data_re)
img.save('reconstructed.jpg')
