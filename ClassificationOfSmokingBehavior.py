import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score, auc,roc_curve,f1_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay#, will be needed: classification_report, confusion_matrix,
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB#,BernoulliNB
from sklearn.neural_network import MLPClassifier

#used to encode catagorical variables into numerical values 
#in this case they are encoded to binary values since they are only two values in each catagorical variable
from sklearn.preprocessing import LabelEncoder,StandardScaler#MinMaxScaler



sns.set_style('darkgrid')
class ClassificationTasks:
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def k_nearest_neighbors(self,num_neighbors=58):
        knn = KNeighborsClassifier(n_neighbors=num_neighbors)
        knn.fit(self.x_train,self.y_train)
        # accuracy = accuracy_score(self.y_test,y_pred)
        return knn

    
    def decision_tree(self):
        #10:.9132231404958677, 42:.9090909090909091
        clfTree = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=40,random_state=10)
        clfTree.fit(self.x_train,self.y_train)
        dot_data = tree.export_graphviz(clfTree,out_file=None,filled=True,rounded=True,special_characters=True,
                                        class_names=['non-Smoker','Smoker'],
                                        feature_names=['Age','Gender','BMI','Region','No. Childred','Insurance Charges'])
        graph = graphviz.Source(dot_data)
        graph.render("Smoking decision tree Model")
        
        # graph.view()
        # print(classification_report(self.y_test, y_pred))
        return clfTree
    
    #since our data attributes have continuous distribution we came to conclusion that the best we could use is gaussianNB
    #normal program execution 
    #BernoulliNB: 0.859504132231405, GaussianNB: 8057851239669421
    # preprocessing step 2 not included (feature scalling) using BernoulliNB:
    # BernoulliNB: 0.7272727272727273, GaussianNB:8057851239669421
    def NaiveBayes(self):
        naiveBayesModel = GaussianNB()
        naiveBayesModel.fit(self.x_train,self.y_train)
        # print('to compare:')
        # print(self.y_test[:10])
        # print(naiveBayesModel.predict(self.x_test[:10]))
        return naiveBayesModel
    
    def neuralNetwork(self):
        #0.001: 0.849862258953168, .003: 0.8650137741046832,
        #.005:0.8746556473829201, 009:0.8760330578512396,.07 and 08:0.8801652892561983,
        #chosen value: .05:0.8815 426997245179
        modelANN = MLPClassifier(hidden_layer_sizes=1,max_iter=500,activation='logistic',random_state=42,learning_rate_init=0.05)
        modelANN.fit(self.x_train,self.y_train)
        # y_pred = modelANN.predict(self.x_test)
        # print (y_pred[:10])
        # print(self.y_test[:10])
        return modelANN
    
class DataPreprocessing:
    def __init__(self,data):
        self.data = data

    #encode gender,smoker and region (catagorical variables into numerical values)
    def preformDataPreprocessingStep1(self):
        self.__encodeCatagoricalVariables(self.data)
        return self.data
    
    def preformDataPreprocessingStep2(self):
        self.__featureScalling(self.data)
        return self.data
    
    def __encodeCatagoricalVariables(self,data):
        #encoding catagorical variables using label Encoder to generate 1,0 
        #dummies: to generate one-hot encoding not used which was not used by us since we have only two values in each catagorical variable
        # and splitting each one into two columns will not be efficient as we thought
        label_Encoder = LabelEncoder()
        #gender encoding
        #male:1, female:0
        data['Gender'] = label_Encoder.fit_transform(data['Gender'])
        #smoker encoding
        #yes:1, no:0
        data['Smoker'] = label_Encoder.fit_transform(data['Smoker'])
        #region encoding (region has only two values in our case that is why we use label binary encoding) and not one-hot encoding
        #south:1, north:0
        data['Region'] = label_Encoder.fit_transform(data['Region'])
        self.data= data
    
    #feature scalling using Standardisation (Z-score normalisation), min max
    def __featureScalling(self,data):
        scaler = StandardScaler() # since standarization outpreformed MinMaxScaler in our case
        scaler.fit(data[['Age','BMI','No. Childred','Insurance Charges']])
        data[['Age','BMI','No. Childred','Insurance Charges']] = scaler.transform(data[['Age','BMI','No. Childred','Insurance Charges']])
        # print(data['No. Childred'].max())

def data_visualization(data):
    # Show the distribution of the class label (Smoker)and indicate any highlights in the distribution of the class label
    # 0 show count of each class
    print(data['Smoker'].value_counts())
    # 1 show the distribution of the class label (Smoker)
    sns.countplot(x="Smoker", data=data)
    plt.title("Distribution of smoker class in the dataset")
    plt.show()

    # 2. show the density plot for the age
    sns.kdeplot(data["Age"],fill=True)
    plt.title("Density plot for the age")
    plt.show()

    # 3. show the density plot for the BMI
    sns.kdeplot(data['BMI'],fill=True)
    plt.title("Density plot for the BMI")
    plt.show()

    # 4. Visualize the scatterplot of the data and split based on Region attribute
    # between age and bmi as they can be the most representive features here
    sns.scatterplot(x="Age", y="BMI", hue="Region",data=data,) #hue used to group variables  based on a categorical variable()
    plt.title("Scatter plot for the Age against BMI splited based on (region)")
    plt.show()

    # Extra corelation matrix to see dependancy between features
    #used to see dependancy between features since that is most important for the classification task of Naive Bayes
    # as it assumes that there are no dependancy between features and that assumption could show the poor performance of the model 
    # in our case it showed that there is dependancy between some features which may cause a poor performance of the model
    corrMat = data.corr(method="pearson")
    cmap = sns.diverging_palette(250, 354,80,60,center='dark', as_cmap=True)
    sns.heatmap(corrMat,vmax=1,vmin=-.5,cmap=cmap,square=True,linewidths=.5,annot=True)#cbar_kws={"shrink": .5},
    plt.title("Corelation matrix between features")
    plt.show()
    # plt.title("Corelation matrix")
    # plt.show()

    # List of continuous features to plot to see their distribution 
    #comment out after finish 
    #----------------------------------------------------------------
    # continuous_features = ['Age', 'BMI', 'Insurance Charges']
    # for feature in continuous_features:
    #     plt.figure(figsize=(10, 4))

    #     # Histogram
    #     plt.subplot(1, 2, 1)
    #     sns.histplot(data=data, x=feature, kde=True)
    #     plt.title(f'Histogram of {feature}')

    #     plt.show()

    #-----------------------------------------------------------------

def splitData(data,test_size=.2,seed=10):
    X = data.drop('Smoker',axis=1)
    y = data['Smoker']
    return train_test_split(X,y,test_size=test_size,random_state=seed,shuffle=True)

def showConfusionMatrix(y_test, y_pred, modelName):
    #plotting the confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[False,True])
    cm_display.plot().ax_.set_title(f'Confusion Matrix for {modelName}')
    plt.show()

def showRoCcurve(fpr,tpr,modelName):
    roc_auc_curve = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_curve))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('(ROC) Curve for model ='+modelName)
    plt.legend(loc="lower right")
    plt.show()
        
def main():
    # Load the data
    data = pd.read_csv("data/Data_project2.csv")
    #Preprossing data : # Encode catagorical variables and feature scalling for numberOfChildren, BMI, Insurance Charges
    dataPreprocessing = DataPreprocessing(data)
    data = dataPreprocessing.preformDataPreprocessingStep1()

    # Data visualization
    data_visualization(data)

    #perform feature scalling on data
    data = dataPreprocessing.preformDataPreprocessingStep2()
    print("---------------------------------")
    print("Data after preprocessing")
    print(data.head())
    print("---------------------------------")
   
    # Split the data into training and testing
    X_train,X_test,y_train,y_test = splitData(data)

    # Classification tasks
    classificationTasks = ClassificationTasks(X_train,X_test,y_train,y_test)
    #creating models
    models ={
        #overfitting
    'KNN (k=10)': classificationTasks.k_nearest_neighbors(5),
    #should be right
    'KNN (k=sqrt(N=59))': classificationTasks.k_nearest_neighbors(),
    #underfitting
    'KNN (k=80)': classificationTasks.k_nearest_neighbors(110),
    'Decision Tree': classificationTasks.decision_tree(),
    'Naive Bayes': classificationTasks.NaiveBayes(),
    'ANN': classificationTasks.neuralNetwork()
    }

    results = {'Model': [], 'Accuracy': [], 'ROC/AUC': [],'Precision':[],'Recall':[],'F1-score':[]}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuaracy = accuracy_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_pred)
        # confusionMatrix = confusion_matrix(y_test,y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        results['Model'].append(name)
        results['Accuracy'].append(accuaracy)
        results['ROC/AUC'].append(roc_auc)
        # results['Confusion Matrix'].append(confusionMatrix)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1-score'].append(f1score)
        #show the roc curve
        fpr, tpr,_ = roc_curve(y_test, y_pred)
        showConfusionMatrix(y_test,y_pred,name)
        showRoCcurve(fpr,tpr,name)

    # Display results in a DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == "__main__":
    main()