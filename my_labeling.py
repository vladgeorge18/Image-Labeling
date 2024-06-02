__authors__ = ['1718986','1719379']
__group__ = 'noneyet'

from utils_data import read_dataset, read_extended_dataset, crop_images, Plot3DCloud,visualize_k_means,visualize_retrieval
from Kmeans import __authors__, __group__, KMeans, distance, get_colors
from KNN import __authors__, __group__, KNN
from PIL import Image
import numpy as np
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    #print(len(train_imgs))
    #print(train_class_labels)
    ##print(classes)
    #print(background[:10])
    #print(class_labels[:10])
    #print(color_labels[:10])
    
    #K=5 
    #image_data = cropped_images[2]
    #image_data1 = imgs[2]

    #print(train_imgs.shape)
    #print(train_class_labels.shape)
    
    
    #print("-------------")
    #print(knn.neighbors)

    #kmeans = KMeans(imgs[2], 4, {'km_init': 'first'})
    #kmeans.fit()
    #print(get_colors(kmeans.centroids),kmeans.get_precentage())

    #kmeans.find_bestK(10)
    #print(kmeans.K)
    #visualize_k_means(kmeans,imgs[2].shape)
    #ax = Plot3DCloud(kmeans)
    #plt.show()

    #print(class_labels[:10])
    #print(color_labels[:20])
    #visualize_retrieval(imgs[:12],12,info =color_labels[:12],title='12 images of differnt colors')
    #visualize_retrieval(cropped_images[:12],12)
    def Retrieval_by_color(images, tags, color_precentage, question_tags, precentage):
        returned_images = []
        for i in range(len(images)):
            precs = []
            add = True
            for qtag in question_tags:
                prec = 0
                for idx, elem in enumerate(tags[i]):
                    if qtag == elem:
                        prec+= color_precentage[i][idx]
                
                precs.append(prec)
            
            for elem1, elem2 in zip(precs,precentage):
                if elem2 > elem1:
                    add = False
                    break
            
            if add == True:
                returned_images.append(images[i])
        
        return returned_images
    '''
    # Testing retrieval by color
    visualize_retrieval(imgs[:12],12,info =color_labels[:12],title='12 images of differnt colors')
    images = []
    tags = []
    color_precentage = []
    for cropped_img,img in zip(cropped_images[:12],imgs[:12]):
        kmeans = KMeans(cropped_img, 3, {'km_init': 'first'})
        kmeans.fit()
        images.append(img)
        tags.append(get_colors(kmeans.centroids))
        color_precentage.append(kmeans.get_precentage())
        print(get_colors(kmeans.centroids),kmeans.get_precentage())

    question_tags = ['Blue']
    returned_imgs = Retrieval_by_color(images,tags,color_precentage,question_tags=question_tags,precentage=[0.7])
    #print(returned_imgs)
    visualize_retrieval(returned_imgs,len(returned_imgs),title='Images retrived with at least 70% blue color')
    '''
    #images = [1,2,3,4,5,6]
    #tags = [['r','y','w'],['b','b','r'],['g','r'],['b','r','g','b'],['t'],['b']]
    #color_precentage = [[0.2,0.3,0.5],[0.5,0.4,0.1],[0.2,0.8],[0.1,0.5,0.1,0.3],[1],[1]]
    #question_tags = ['r']
    #precentage = [0.7]

    #print(Retrieval_by_color(images, tags, color_precentage, question_tags, precentage))

    def Retrieval_by_shape(images, tags, knn_neighbors, question_tag, precentage):
        returned_images = []
        for i in range(len(images)):
            
            if tags[i] == question_tag:
                count = 0
                for neighbor in knn_neighbors[i]:
                    if neighbor == question_tag:
                        count+=1
                prec = count / len(knn_neighbors[i])
                print(prec)
                print(knn_neighbors[i])
                if prec >= precentage:
                    returned_images.append(images[i])
        
        return returned_images
    
    #images = [1,2,3,4,5,6]
    ''''
    tags = ['Heels', 'Sandals', 'Jeans', 'Shorts', 'Jeans', 'Flip Flops']
    knn_neighbors = [['Heels', 'Heels', 'Heels', 'Heels', 'Heels', 'Heels', 'Heels'],
 ['Sandals', 'Sandals', 'Flip Flops', 'Flip Flops', 'Sandals', 'Sandals',
  'Flip Flops'],
 ['Jeans', 'Jeans', 'Jeans', 'Jeans', 'Jeans', 'Jeans', 'Jeans'],
 ['Shorts', 'Shorts', 'Shorts', 'Shorts', 'Jeans', 'Shorts', 'Jeans'],
 ['Jeans', 'Jeans', 'Jeans', 'Jeans', 'Jeans', 'Jeans', 'Jeans'],
 ['Flip Flops', 'Sandals', 'Flip Flops', 'Sandals', 'Flip Flops', 'Flip Flops',
  'Flip Flops'] ]
    '''
    #question_tag = 'Sandals'
    #precentage = 0.55
    
    #Testing retrieval_by_shape
    ''''
    knn = KNN(train_imgs, train_class_labels)
    knn._init_train(train_imgs)
    res = knn.predict(test_imgs[:12],7)

    print(res)
    print(test_class_labels[:12])

    returned_images = Retrieval_by_shape(test_imgs[:12], res, knn.neighbors, question_tag='Heels', precentage=0.4)
    visualize_retrieval(test_imgs[:12],12,info = test_class_labels[:12],title='12 images of differnt shapes')
    visualize_retrieval(returned_images,len(returned_images),title='Images retrived with at least 40% percentage as Heels')

    '''

    def Retrieval_combined(images,shape_tags,color_tags,question_shape,question_color,my_color_precentage,my_shape_precentage,knn_neighbors,kmeans_color_precentage):
        returned_images = []
        for i in range(len(images)):
            #shape precentage
            if question_shape == shape_tags[i]:
                count = 0
                for neighbor in knn_neighbors[i]:
                    if neighbor == question_shape:
                        count+=1
                
                prec = count / len(knn_neighbors[i])

                if prec >= my_shape_precentage:
                    #do the same for color precentage
                    
                    precs = []
                    add = True
                    for qtag_color in question_color:
                        prec_color = 0
                        for idx, elem in enumerate(color_tags[i]):
                            if qtag_color == elem:
                                prec_color += kmeans_color_precentage[i][idx] 
                    
                        precs.append(prec_color)
                    
                    for elem1,elem2 in zip(precs,my_color_precentage):
                        if elem2 > elem1:
                            add = False
                            break
                    
                    if add == True:
                        returned_images.append(images[i])

        return returned_images

    ''''
    print(len(imgs))
    print(len(train_imgs))
    print(len(test_imgs))
    '''
    #Testing retrieval combined
    
    '''
    labels =[]
    for shape, colors_l in zip(class_labels,color_labels):
        string =''
        colors = ''
        for col in colors_l:
            colors+= col+" "
        string = shape +" with color/s: "+colors
        labels.append(string)

    #print(labels[:10])
    visualize_retrieval(imgs[:16],16,info=labels[:16],title='16 images of differnt shapes and colors')
    
    knn = KNN(train_imgs, train_class_labels)
    knn._init_train(train_imgs)
    shape_tags = knn.predict(imgs[:16],7)
    my_shape_precentage = 0.7
    knn_neighbors = knn.neighbors
    #print(shape_tags)
    images = []
    color_tags = []
    kmeans_color_precentage = []

    for cropped_img,img in zip(cropped_images[:16],imgs[:16]):
        kmeans = KMeans(cropped_img, 3, {'km_init': 'first'})
        kmeans.fit()
        images.append(img)
        color_tags.append(get_colors(kmeans.centroids))
        kmeans_color_precentage.append(kmeans.get_precentage())
        #print(get_colors(kmeans.centroids),kmeans.get_precentage())
    
    question_shape = 'Dresses'
    question_color = ['Black']
    my_color_precentage = [0.7]
    res = Retrieval_combined(images,shape_tags,color_tags,question_shape,question_color,my_color_precentage,my_shape_precentage,knn_neighbors,kmeans_color_precentage)

    visualize_retrieval(res,len(res),title='Images retrieved for Black Dresses')

    '''
    def Kmean_statistics(images, labels, max_k):
        wcds = []
        iters = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for k in range(2, max_k):
            print(f'K={k}')
                        
            avg_wcds = []
            avg_iters = []
            preds = []
            
            for idx, img in enumerate(images):
                km = KMeans(img, k,{'km_init': 'first'})
                km.fit()
                centroids = km.centroids
                color = get_colors(centroids)
                
                preds.append(color)
                                
                avg_wcds.append(km.withinClassDistance())
                avg_iters.append(km.iter)
                
            wcds.append(sum(avg_wcds)/len(avg_wcds))
            iters.append(sum(avg_iters)/len(avg_iters))
            f1,prec,rec = Get_color_accuracy(preds, labels)
            f1_scores.append(f1)
            precision_scores.append(prec)
            recall_scores.append(rec)
                
            print(f'Average within class distance for k={k}: {sum(avg_wcds)/len(avg_wcds)}')
            print(f'Average iterations for k={k}: {sum(avg_iters)/len(avg_iters)}')
                        

        print(f'WCDS Score: {wcds}')
        print(f'F1 Score: {f1_scores}')
        print(f'Precison Score: {precision_scores}')  
        print(f'Recall Score: {recall_scores}')        

        # Make 2 subplots
        fig, axs = plt.subplots(1, 5,figsize=(25, 7))
        fig.suptitle('Kmean statistics')
        axs[0].plot(range(2, max_k), wcds)
        axs[0].set_title('Average within class distance')
        axs[0].set(xlabel='K', ylabel='WCD')
        axs[1].plot(range(2, max_k), iters)
        axs[1].set_title('Average iterations')
        axs[1].set(xlabel='K', ylabel='Iterations')
        axs[2].plot(range(2, max_k), f1_scores)
        axs[2].set_title('F1 Score')
        axs[2].set(xlabel='K', ylabel='F1 Score')
        axs[3].plot(range(2, max_k), precision_scores)
        axs[3].set_title('Precision Score')
        axs[3].set(xlabel='K', ylabel='Precision Score')
        axs[4].plot(range(2, max_k), recall_scores)
        axs[4].set_title('Recall Score')
        axs[4].set(xlabel='K', ylabel='Recall Score')
        plt.subplots_adjust(wspace=0.5)
        plt.show()
            
    def KNN_statistics(knn, test_imgs, test_class_labels, max_k):
        accuracy = []
        
        for k in range(1, max_k):
            print(f'K={k}')
            
            # knn = KNN(images, labels)
            preds = knn.predict(test_imgs, k)
                        
            accuracy.append(Get_shape_accuracy(preds, test_class_labels))
                        
        # Plot accuracy
        plt.plot(range(1, max_k), accuracy)
        plt.title('KNN with new features and cosine distance accuracy')
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.show()
            
    def Get_shape_accuracy(predictions, labels):
        correct = 0
        for i in range(len(predictions)):
            
            if predictions[i] == labels[i]:
                correct += 1
                
        return correct/len(predictions)
    
    # Returns the F1 score
    def Get_color_accuracy(predictions, labels):
        recalls = []
        precisions = []
        
        for i in range(len(predictions)):
            True_positives = 0
            
            pred = set(predictions[i])
            
            for elem in pred:                
                if elem in labels[i]:
                    True_positives += 1
                    
            if len(labels[i]) != 0:
                recall = True_positives/len(labels[i])
                precision = True_positives/len(pred)
            else:
                continue
            
            recalls.append(recall)
            precisions.append(precision)
        
        precisions = sum(precisions)/len(precisions)
        recalls = sum(recalls)/len(recalls)
        f1 = 2*precisions*recalls/(precisions+recalls)
        return f1 , precisions, recalls
            
    #Kmean_statistics(cropped_images, color_labels, 10)
    
    
    import hashlib

    
    def hash_image(img):
        return hashlib.sha256(img.tobytes()).hexdigest()

    #train_hashes = set(hash_image(img) for img in train_imgs)

    # Check how many images from the test set are in the training set
    #common_images_count = sum(1 for img in test_imgs if hash_image(img) in train_hashes)

    #print(f"Number of images in the test set that are also in the training set: {common_images_count}")
    #print(len(test_imgs))
    
    # Create a new training set by filtering out images that are in the test set
    '''
    new_test_imgs = []
    new_test_labels = []

    for img, label in zip(test_imgs, test_class_labels):
        if hash_image(img) not in train_hashes:
            new_test_imgs.append(img)
            new_test_labels.append(label)

    new_test_imgs = np.array(new_test_imgs)
    new_test_labels = np.array(new_test_labels)
    
    '''
    #knn = KNN(train_imgs, train_class_labels,add_features = True)
   
    #cosine_knn = KNN(train_imgs, train_class_labels, 'cosine', add_features=True)
    

    #KNN_statistics(knn, new_test_imgs, new_test_labels, 10)    
    #KNN_statistics(cosine_knn, new_test_imgs, new_test_labels, 10)
'''


# plot wcd comparison

k_values = range(2, 10)

hypercube_kmeans_wcds = [1387.6045115871, 653.7039777455658, 414.99076006265045, 299.00912500795175, 235.621528317196, 193.67556678153346, 167.26076223994164, 147.81000690673838]
kmeans_plus_plus_wcds = [1383.5102382699959, 654.5671093746192, 416.5527163301148, 299.2956313949226, 233.7670436143071, 191.08053823205148, 159.76702062348772, 138.65633894249657]
baseline_kmeans_wcds = [1386.4722737711181, 679.6528571362741, 419.83029582800634, 307.4296386611924, 242.28772604964033, 206.6550105118784, 181.7581393584416, 163.59320669671675]

plt.figure(figsize=(10, 6))

# Plot each WCDS score list
plt.plot(k_values, hypercube_kmeans_wcds, marker='o', linestyle='-', label='Hypercube K-means')
plt.plot(k_values, kmeans_plus_plus_wcds, marker='o', linestyle='-', label='K-means++')
plt.plot(k_values, baseline_kmeans_wcds, marker='o', linestyle='-', label='Baseline K-means')

# Add titles and labels
plt.title('Comparison of WCDS Scores for Different K-means Methods')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCDS Score')

plt.ylim(100, 1500)  # Set the y-axis range to focus on the scores
plt.yticks(range(100, 1600, 100))  # Set y-axis ticks to be more representative

plt.legend()

# Show the plot
plt.show()
'''