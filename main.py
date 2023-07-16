import os
import cv2
import numpy as np
import pandas as pd
from keras.applications.xception import Xception, preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('./data.csv', sep=";")

# Rename columns using a dictionary
df = df.rename(columns={'ams_id_ams': 'Advertisement ID', 'ams_rep': 'Reporter Count', 'ams_nom_img': 'Image Name', 'ams_site': "Site", 'ams_url_annonce': 'Advertisement URL', 'ams_num_enreg_txt': 'Registration Number'})


def getImagePaths(path):
    """
    Function to Combine Directory Path with individual Image Paths
    
    parameters: path(string) - Path of directory
    returns: image_names(string) - Full Image Path
    """
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names

def preprocess_img(img_path):
    dsize = (225,225)
    new_image=cv2.imread(img_path)
    new_image=cv2.resize(new_image,dsize,interpolation=cv2.INTER_NEAREST)  
    new_image=np.expand_dims(new_image,axis=0)
    new_image=preprocess_input(new_image)
    return new_image

def load_data():
    output=[]
    output=getImagePaths(images_dir)[:10000]
    return output

def model():
    model=Xception(weights='imagenet',include_top=False)
    for layer in model.layers:
        layer.trainable=False
        #model.summary()
    return model

def feature_extraction(image_data,model):
    features = model.predict(image_data)
    features = np.array(features)
    features = features.flatten()
    return features

def result_vector_cosine(model, feature_vector, new_img):
    new_feature = model.predict(new_img)
    new_feature = np.array(new_feature)
    new_feature = new_feature.flatten()
    N_result = 12
    nbrs = NearestNeighbors(n_neighbors=N_result, metric="cosine").fit(feature_vector)
    distances, indices = nbrs.kneighbors([new_feature])
    
    # Calculate similarity scores
    similarity_scores = 1 - distances
    max_similarity = similarity_scores.max()
    similarity_scores_normalized = similarity_scores/max_similarity
    
    return similarity_scores_normalized, indices

def input_show(data):
    plt.title("Query Image")
    plt.imshow(data)

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_result(data,result):
    fig = plt.figure(figsize=(12,8))
    for i in range(0,12):
        index_result=result[0][i]
        plt.subplot(3,4,i+1)
        # Read the image using OpenCV in BGR format
        bgr_image = cv2.imread(data[index_result])
        # Convert BGR image to RGB
        rgb_image = bgr_to_rgb(bgr_image)
        plt.imshow(rgb_image)
    plt.show()

if __name__=="__main__":
    threshold = 0.75
    images_dir = './Nouveau dossier/'

    output=load_data()
    main_model=model()
    features=[]

    print("Training Model")
    #Limiting the data for training
    for i in output[:1008]:
        new_img=preprocess_img(i)
        features.append(feature_extraction(new_img,main_model))
    feature_vec = np.array(features)
    
    matching_df = pd.DataFrame(columns=['Reg No 1', 'Reg No 2', 'Ad Num 1', 'Ad Num 2','Image Number 1',"Image Number 2" ,'Similarity Score'])
    

    print("Querying Model")
    for index, row in df.iterrows():
        print(row)
        # The path of the image whose entry is queryed
        output_path=f"{images_dir}{row['registration_number']}/{row['Advertisement ID']}/photo_{row['Image Name']}"
        try:
            distances, indices = result_vector_cosine(main_model,feature_vec,preprocess_img(output_path))
        except Exception as e:
            continue

        filtered_path = []
        for dist,index in zip(distances[0], indices[0]):
            if(dist>threshold):
                filtered_path.append((output[index],dist))
        
        for file_path, score in filtered_path:
            # Extract the file name from the path
            file_name = os.path.basename(file_path)
            # Remove the "photo_" prefix
            image_name = file_name.replace("photo_", "")
            # Extract the advertisement number from the image name
            advertisement_number = file_path.split("/")[3]
            # Extract the advertisement number from the image name
            reg_number = file_path.split("/")[2]

            # Add the matching image information to the DataFrame
            matching_df = matching_df.append({
                'Reg No 1': row['registration_number'],
                'Reg No 2': reg_number,
                'Image Number 1': row['Image Name'] ,
                'Image Number 2': image_name ,
                'Ad Num 1': row['Advertisement ID'],
                'Ad Num 2': advertisement_number,
                'Similarity Score': score
            }, ignore_index=True)


    count_df = matching_df.groupby(['Ad Num 1', 'Ad Num 2','Reg No 1','Reg No 2']).size().reset_index(name='Count')
    
    # Convert 'Ad Num 1' and 'Ad Num 2' to strings
    count_df['Ad Num 1'] = count_df['Ad Num 1'].astype(str)
    count_df['Ad Num 2'] = count_df['Ad Num 2'].astype(str)
    
    # Filter rows based on 'Count' greater than 10 and 'Ad Num 1' is different from 'Ad Num 2'
    filtered_count_df = count_df[(count_df['Ad Num 1'] != count_df['Ad Num 2'])]
    filtered_count_df[['Ad Num 1', 'Ad Num 2']] = np.sort(filtered_count_df[['Ad Num 1', 'Ad Num 2']], axis=1)
    # Drop duplicates based on the sorted values of 'Ad Num 1' and 'Ad Num 2'
    filtered_count_df.drop_duplicates(subset=['Ad Num 1', 'Ad Num 2'], keep='first', inplace=True)
    
    # Display the updated DataFrame
    print(filtered_count_df)
    
    # Save the updated DataFrame to a CSV file
    filtered_count_df.to_csv('filtered_count_df.csv', index=False)
    # Display a message to indicate successful saving
    print("DataFrame saved to 'filtered_count_df.csv'")
