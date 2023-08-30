#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <array>
#include <string>
#include <cmath>
#include <algorithm>   
#include <numeric>                      

/*VECTORIZATION SON*/

/*possible major updates: MYSQL for data I/O, eigen for
operations on matrices, openCL to use GPU acceleration, 
multithreading*/

#ifndef VECTORMINE
#define VECTORMINE

/*this namespace provides several functions that are responsible for
conducting various operation on vectors. now main priority is to change 
importing vectors to importing pointers to vector. remember that it would
require creating resultVector every time.*/

namespace vector{

/*this function creates new vector which is part of provided vector.
both start and end indexes starts at 0*/

template<typename T> std::vector<T> vectorSlicer(std::vector<T> oldVector, int startIndex, int endIndex){
    auto start = oldVector.begin() + startIndex;
    auto end = oldVector.begin() + endIndex + 1;
    std::vector<T> newVector(start, end);
    return newVector;
}

/*element-wise multiplication of two vectors. It means vector1 = {a0, a1 ... an}
vector2 = {b0, b1 ... bn} vectorResult = {a0*b0, a1*b1 ... an*bn}*/

template<typename T> std::vector<T> vectorElementWiseMultiplication(std::vector<T> firstVector, std::vector<T> secondVector){
    std::vector<T> vectorResult(firstVector.size());
    std::transform(firstVector.begin(), firstVector.end(), secondVector.begin(), vectorResult.begin(), [](T a, T b) {return a * b;});
    return vectorResult;
}

/*element-wise division of two vectors. It means vector1 = {a0, a1 ... an}
vector2 = {b0, b1 ... bn} vectorResult = {a0/b0, a1/b1 ... an/bn}*/

template<typename T> std::vector<T> vectorElementWiseDivision(std::vector<T> firstVector, std::vector<T> secondVector){
    std::vector<T> vectorResult(firstVector.size());
    std::transform(firstVector.begin(), firstVector.end(), secondVector.begin(), vectorResult.begin(), [](T a, T b) {return a / b;});
    return vectorResult;
}

/*addition of two vectors. It means vector1 = {a0, a1 ... an}
vector2 = {b0, b1 ... bn} vectorResult = {a0+b0, a1+b1 ... an+bn}*/

template<typename T> std::vector<T> vectorAddition(std::vector<T> firstVector, std::vector<T> secondVector){
    std::vector<T> vectorResult(firstVector.size());
    std::transform(firstVector.begin(), firstVector.end(), secondVector.begin(), vectorResult.begin(), [](T a, T b) {return a + b;});
    return vectorResult;
}

/*subtraction of two vectors. It means vector1 = {a0, a1 ... an}
vector2 = {b0, b1 ... bn} vectorResult = {a0-b0, a1-b1 ... an-bn}*/

template<typename T> std::vector<T> vectorSubtraction(std::vector<T> firstVector, std::vector<T> secondVector){
    std::vector<T> vectorResult(firstVector.size());
    std::transform(firstVector.begin(), firstVector.end(), secondVector.begin(), vectorResult.begin(), [](T a, T b) {return a - b;});
    return vectorResult;
}

/*this function multiplies every element of a vector by a scalar*/

template<typename T> std::vector<T> vectorScalarMultiplication(std::vector<T> firstVector, T scalar){
    std::transform(firstVector.begin(), firstVector.end(), firstVector.begin(), [scalar](T value) {return value * scalar;});
    return firstVector;
}

/*this function has two modes: in first, when bool mode == true it
devides every value in provided vector by a scalar. if mode == false
it creates changes vector.at(i) to scalar / vector.at(i)*/

template<typename T> std::vector<T> vectorScalarDivision(std::vector<T> firstVector, T scalar, bool mode /*true - divides vector by scalar, false - divides scalar by vector*/){
    std::transform(firstVector.begin(), firstVector.end(), firstVector.begin(), [scalar, mode](T value) {return mode ? (value / scalar) : (scalar / value);});
    return firstVector;
}

/*it simply calcuates exp() function for every element in the vector*/

template<typename T> std::vector<T> vectorExp(std::vector<T> firstVector){
    std::vector<T> resultVector(firstVector.size());
    std::transform(firstVector.begin(), firstVector.end(), resultVector.begin(), static_cast<T (*)(T)>(std::exp));
    return resultVector;
}

/*simply adds a number to every value in vector*/

template<typename T> std::vector<T> vectorScalarAddition(std::vector<T> firstVector, T number){
    std::transform(firstVector.begin(), firstVector.end(), firstVector.begin(), [number](T value) { return value + number; });
    return firstVector;
}

/*it works in two mods. in first, when bool mode == true it subtract
a number form every value of vector. if bool == false tehn it changes
vector.at(i) to scalar - vector.at(i)*/

template<typename T> std::vector<T> vectorScalarSubtraction(std::vector<T> firstVector, T scalar, bool mode /*true - substract scalar from vector, false - substract vector from scalar*/){
    std::transform(firstVector.begin(), firstVector.end(), firstVector.begin(), [scalar, mode](T value) {return mode ? (value - scalar) : (scalar - value);});
    return firstVector;
}

/*it changes every vector inside of dataMatrix to a scalar by summing its elements
, which creates a vector of sums. then this vector is element-wise multipied
by data vector, and it's a result*/

template<typename T> std::vector<T> vectorMatrixElementWiseMultipication(std::vector<T> dataVector, std::vector<std::vector<T>> dataMatrix){
    auto dot_product = [](const std::vector<T>& v1, const std::vector<T>& v2) {
        return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0);
    };
    std::vector<T> result(dataVector.size());
    std::transform(dataMatrix.begin(), dataMatrix.end(), result.begin(), [&](const std::vector<T>& row) {
        return dot_product(dataVector, row);
    });
    return result;
}

/*it just multipies every element it one matrix by corresponding element
of second matrix*/

template<typename T> std::vector<std::vector<T>> matrixElementWiseMultiplication(std::vector<std::vector<T>> firstMatrix, std::vector<std::vector<T>> secondMatrix){
    std::vector<std::vector<T>> matrixResult(firstMatrix.size());
    std::transform(firstMatrix.begin(), firstMatrix.end(), secondMatrix.begin(), matrixResult.begin(), [](std::vector<T> a, std::vector<T> b) {return vectorElementWiseMultiplication(a, b);});
    return matrixResult;
}

}; //end of vector namespace
#endif //VECTORMINE

#ifndef EXCEPTION
#define EXCEPTION
class Exception : public std::exception {
private:
    std::string errorMessage;

public:
    Exception(const std::string& message) : errorMessage(message) {}

    const char* what() const noexcept override {
        return errorMessage.c_str();
    }
};
#endif //EXCEPTION

#ifndef LOADDATA
#define LOADDATA

namespace LoadData{

template<typename T> struct loadFileData{
public:
    std::vector<int> config;
    std::vector<std::vector<T>> dataVector;
    std::vector<std::vector<T>> weightsVector;
    std::vector<T> solutionsVector;
    std::vector<std::vector<T>> biasesVector;
};

template<class T> class loadAllData{
private:
    std::vector<int> loadConfig(std::string configL){
        std::vector<int> tempData;
        std::ifstream stream;
        stream.open(configL);
        try{
            if (stream.is_open()){
                int tempInt = 0;
                stream >> tempInt;
                tempData.push_back(tempInt);
                stream >> tempInt;
                tempData.push_back(tempInt);
                std::vector<int> tempVector(tempData.at(1), 0);
                for (int i = 0; i < 2; i++){
                    for(int j = 0; j < tempInt; j++){
                    stream >> tempVector[j];
                    }
                    tempData.insert(tempData.end(), tempVector.begin(), tempVector.end());
                }
            }else{throw Exception("Error: unable to open file");}
        }
        catch(const Exception& e){
            std::cout << "Exception caught: " << e.what() << "file name: "<< configL << std::endl;
            std::exit(1); //możnaby jakoś te kody do basha przekazywać
        }
        stream.close();
        return tempData;
    }

    std::vector<std::vector<T>> loadDataVector(std::string dataL, int numberOfExamples, int exampleLenght){
        std::vector<std::vector<T>> tempData(numberOfExamples);
        std::ifstream stream;
        stream.open(dataL);
        try{
            if (stream.is_open()){
                T temp;
                for (int i = 0; i < numberOfExamples; i++){
                    for (int j = 0; j < exampleLenght; j++){
                    temp = 0;
                    stream >> temp;
                    tempData.at(i).push_back(temp);
                    }
                }
            }else{throw Exception("Error: unable to open file");}
        }
        catch(const Exception& e){
            std::cout << "Exception caught: " << e.what() << "file name: "<< dataL << std::endl;
            std::exit(1);
        }
        stream.close();
        return tempData;
    }

    std::vector<std::vector<std::vector<T>>> loadWeightsVector(std::string weightsL, int numberOfLayers, std::vector<T> neuronsPerLayer){
        std::vector<std::vector<std::vector<T>>> tempWeights;
        std::vector<std::vector<T>> tempVector2Dim;
        std::vector<T> tempVector1Dim;
        std::ifstream stream;
        stream.open(weightsL);
        try{
            if (stream.is_open()){
                T temp;
                for(int i = 0; i < (numberOfLayers - 1); i++){
                    for(int j = 0; j < neuronsPerLayer.at(i + 1); j++){
                        for(int k = 0; k < neuronsPerLayer.at(i); k++){
                            stream >> temp;
                            tempVector1Dim.push_back(temp);
                        }
                        tempVector2Dim.push_back(tempVector1Dim);
                        tempVector1Dim.clear();
                    }
                    tempWeights.push_back(tempVector2Dim);
                    tempVector2Dim.clear();
                }
            }else{throw Exception("Error: unable to open file");}
        }
        catch(const Exception& e){
            std::cout << "Exception caught: " << e.what() << "file name: "<< weightsL << std::endl;
            std::exit(1);
        }
        stream.close();
        return tempWeights;
    }

    std::vector<std::vector<T>> loadBiasesVector(std::string biasesL, std::vector<T> neuronsPerLayer, int numberOfLayers){
        std::vector<std::vector<T>> tempBiases;
        std::vector<T> tempVector;
        std::ifstream stream;
        stream.open(biasesL);
        try{
            if (stream.is_open()){
                T temp;
                for (int i = 0; i < (numberOfLayers - 1); i++){
                    for (int j = 0; j < neuronsPerLayer.at(i + 1); j++){
                        stream >> temp;
                        tempVector.push_back(temp);
                    }
                    tempBiases.push_back(tempVector);
                    tempVector.clear();
                }
            }else{throw Exception("Error: unable to open file");}
        }
        catch(const Exception& e){
            std::cout << "Exception caught: " << e.what() << "file name: "<< biasesL << std::endl;
            std::exit(1);
        }
        stream.close();
        return tempBiases;
    }

    std::vector<T> loadSolutionsVector(std::string solutionsL, int numberOfExamples){
        std::vector<T> tempSolutions;
        std::ifstream stream;
        stream.open(solutionsL);
        try{
            if (stream.is_open()){
                T temp;
                for (int i = 0; i < numberOfExamples; i++){
                    temp = 0;
                    stream >> temp;
                    tempSolutions.push_back(temp);
                }
            }else{throw Exception("Error: unable to open file");}
        }
        catch(const Exception& e){
            std::cout << "Exception caught: " << e.what() << "file name: "<< solutionsL << std::endl;
            std::exit(1);
        }
        stream.close();
        return tempSolutions;
    }

public:
    loadFileData<T> start(std::string configL, std::string dataL, std::string weightsL, std::string solutionsL,  std::string biasesL){
        loadFileData<T> local;
        local.config = loadConfig(configL);
        local.dataVector = loadDataVector(dataL, local.config.at(0), local.config.at(2));
        local.weightsVector = loadWeightsVector(weightsL, local.config.at(1), vector::vectorSlicer<T>(local.config, 2, (2 + local.config.at(1))));
        local.solutionsVector = loadSolutionsVector(solutionsL, local.config.at(0));
        local.biasesVector = loadBiasesVector(biasesL, vector::vectorSlicer<T>(local.config, 2, (2 + local.config.at(1))), local.config.at(1));
        return local;
    }
};

template <typename T> loadFileData<T> loadNewData(std::string configL, std::string dataL, std::string weightsL, std::string solutionsL, std::string biasesL){
    std::unique_ptr<loadAllData<T>> newLoadFile = std::make_unique<loadAllData<T>>();
    loadFileData<T> returnData = newLoadFile->start(configL, dataL, weightsL, solutionsL, biasesL);
    return returnData;
}
};
#endif //LOADDATA

#ifndef STARTMENU
#define STARTMENU
namespace startmenu{
struct startMenuConfig{
public:
    short mode; //0 - manual, 1 -  auto
    std::array<std::string, 4> locs; // config data weights targetValues
};

class startMenu{
private:
    void embError(void){
        std::cout << "wrong mode code \n";
        std::exit(2);
    }
    std::string manualReader(std::string messange){
        std::cout << " \n where is the " << messange << "?\n";
        std::string tempString = "";
        std::cin >> tempString;
        return tempString;
    }
    startMenuConfig embManual(void){
        std::array<std::string, 4> locs;
        locs.at(0) = manualReader("config");
        locs.at(1) = manualReader("data");
        locs.at(2) = manualReader("weights");
        locs.at(3) = manualReader("target values");
        return {0, locs};
    }
public:
    startMenuConfig startManual(void){
        startMenuConfig config = embManual();
        return config;
    }
    startMenuConfig startAuto(std::array<std::string, 4> locs){
        startMenuConfig config = {1, locs};
        return config;
    }
    void startError(void){
        embError();
        return;
    }
};

startMenuConfig starter(std::string setup, std::array<std::string, 4> locs = {"", "", "", ""}){
    startMenuConfig newConfig;
    newConfig.mode = (short)stoi(setup);
    std::unique_ptr<startMenu> newMenu = std::make_unique<startMenu>();
    switch(newConfig.mode){
    case 0:
        return newMenu->startManual();
        break;
    case 1:
        return newMenu->startAuto(locs);
        break;
    default:
        newMenu->startError();
        break;
    }
    return {newConfig.mode, locs};
}

};
#endif //STARTMENU

#ifndef DATALOADINGCONNECTOR
#define DATALOADINGCONNECTOR
namespace DataLoadingConnector{

template <class T> class connector{
private:
public:
    LoadData::loadFileData<T> start(std::string setup, std::array<std::string, 4> locs = {"", "", "", ""}){
        startmenu::startMenuConfig newConfig = startmenu::starter(setup, locs);
        LoadData::loadFileData newData = LoadData::loadNewData<T>(newConfig.locs.at(0), newConfig.locs.at(1), newConfig.locs.at(2), newConfig.locs.at(3));
        return newData;
    }
};

template <typename T> LoadData::loadFileData<T> startConnector(std::string setup, std::array<std::string, 4> locs = {"", "", "", ""}){
    connector<T> newConnector;
    return newConnector.start(setup, locs);
}
};
#endif //DATALOADINGCONNECTOR

#ifndef ACTIVATIONFUNCTIONS
#define ACTIVATIONFUNCTIONS

/*this namespace provides all activation functions and their derivatives
needed to implement a neural network. Still there is some space for improve
like vectorization of Relu/leakyReLU functions, getting rid of this switch and 
edge cases handling*/

namespace ActivFunctions{

/*this class basicly provides all functionalities mentioned in namespace comment
expect of start function that connects activation functions to other places*/

template<class T> class activationFunctions{
private:

    /*basicly a function that provides calculating whole vector of ReLU functions
    mayby it would be better if those functions were implemented as lambda expressions*/

    std::vector<T> ReLU(std::vector<T> logit){
        std::vector<T> returnVector;
        for(long unsigned int i = 0; i < logit.size(); i++){
            returnVector.push_back(std::max(T(0), logit.at(i)));
        }
        return returnVector;
    }

    /*leaky ReLU implementation, like in ReLU function there is a space for improvement,
    especially getting rid of for loop*/

    std::vector<T> leakyReLU(std::vector<T> logit){
        std::vector<T> returnVector;
        for(long unsigned int i = 0; i < logit.size(); i++){
            (logit.at(i) > static_cast<T>(0)) ? returnVector.push_back(logit.at(i)) : returnVector.push_back((static_cast<T>(0.01) * logit.at(i)));
        }
        return returnVector;
    }

    /*sigmoid activation function. consider creating lambda of inner function
    for better readability*/

    std::vector<T> sigmoid(std::vector<T> logit){
        std::vector<T> returnVector;
        returnVector = vector::vectorScalarDivision(vector::vectorScalarAddition((vector::vectorExp((vector::vectorScalarMultiplication(logit, T(-1))))), T(1)), T(1), false);
        return returnVector;
    }

    /*tanh activation function. consider using STL tanh implementation
    for better effectivness*/

    std::vector<T> tanh(std::vector<T> logit){
        std::vector<T> returnVector;
        returnVector = vector::vectorElementWiseDivision(vector::vectorSubtraction(vector::vectorExp(logit), vector::vectorExp(vector::vectorScalarMultiplication(logit, T(-1)))), vector::vectorAddition(vector::vectorExp(logit), vector::vectorExp(vector::vectorScalarMultiplication(logit, T(-1)))));
        return returnVector;
    }

    /*linear function implementation. is this even needed, as it only returns provided data.*/

    std::vector<T> linear(std::vector<T> logit){
        return logit;
    }

    /*derivative of ReLU function. it needs vectorization instead
    of using for loop*/

    std::vector<T> dReLU(std::vector<T> logit){
        std::vector<T> returnVector;
        for(long unsigned int i = 0; i < logit.size(); i++){
            logit.at(i) >= 0 ? returnVector.push_back(1) : returnVector.push_back(0);
        }
        return returnVector;
    }

    /*derivative of leaky ReLU function. as well as dReLU it needs some vectorization*/

    std::vector<T> dLeakyReLU(std::vector<T> logit){
        std::vector<T> returnVector;
        for(long unsigned int i = 0; i < logit.size(); i++){
            logit.at(i) >= 0 ? returnVector.push_back(1) : returnVector.push_back(0.01);
        }
        return returnVector;
    }

    /*derivative of sigmoid activation function*/

    std::vector<T> dSigmoid(std::vector<T> logit){
        std::vector<T> returnVector;
        returnVector = vector::vectorElementWiseMultiplication(sigmoid(logit), vector::vectorScalarSubtraction(logit, T(1), false));
        return returnVector;
    }

    /*derivative of tanh function*/

    std::vector<T> dTanh(std::vector<T> logit){
        std::vector<T> returnVector;
        returnVector = vector::vectorScalarSubtraction(vector::vectorElementWiseMultiplication(tanh(logit), tanh(logit)), T(1), false);
        return returnVector;
    }

    /*derivative of linear function, consider only importing logit.size()*/

    std::vector<T> dLinear(std::vector<T> logit){
        std::vector<T> returnVector(logit.size(), 1);
        return returnVector;
    }

public:

    /*this function connects all activation functions and their derivatives. 
    first of all, there should be an error-handling mechanism that would check if mode 
    is proper. secondly try to find a way to use it without switch, lastly
    default return of this function shouldn't look like that
    when it comes to bool functionOrDerivative true means function, and false - derivative
    short mode is: 0 - ReLU, 1 - leaky ReLU, 2 - sigmoid, 3 - tanh, 4 - linear.
    remember that 5 is check for data and while implementing error handling 
    use another error messange if this code is 5.*/

    std::vector<T> functionSwitch(std::vector<T> logit, short mode, bool functionOrDerivative){
        switch(mode){
        case 0:
            return functionOrDerivative ? ReLU(logit) : dReLU(logit);
            break;
        case 1:
            return functionOrDerivative ? leakyReLU(logit) : dLeakyReLU(logit);
            break;
        case 2:
            return functionOrDerivative ? sigmoid(logit) : dSigmoid(logit);
            break;
        case 3:
            return functionOrDerivative ? tanh(logit) : dTanh(logit);
            break;
        case 4:
            return functionOrDerivative ? linear(logit) : dLinear(logit);
            break;
        default:
            std::exit(3);
            break;
        }
        return {1};
    }
};

/*start fuction establishes connection between activation functions
and every other functions that call it. it also handles creating class to access those functions.
short mode, and bool functionOrDerivative means the same as in activationFunctions.functionSwitch()*/

template<typename T> std::vector<T> start(std::vector<T> logit, short mode, bool functionOrDerivative){
    activationFunctions<T> newFunction;
    return newFunction.functionSwitch(logit, mode, functionOrDerivative);
}

}; //end of ActivFunctions namespace
#endif //ACTIVATIONFUNCTIONS

#ifndef FORWARDPROPAGATION
#define FORWARDPROPAGATION

/*this namespace provides data structures and functions needed 
to implement forward propagation. still, it needs to be implemented
by other function that combines it with backpropagartion etc. 
to get it fully functionable*/

namespace ForPropagation{

/*this struct contains cache data of single neuron, trained on n examples.*/

template<typename T> struct cacheSingle{
public:
    std::vector<T> logit;
    std::vector<std::vector<T>> weights;
    std::vector<T> biases;
};

/*this struct contains both cache data of a single neuron, and all
activations of this neuron*/

template<typename T> struct forpropagationDataSingle{
public:
    std::vector<T> activations;
    cacheSingle<T> cacheData;
};

/*this is struct of cache data for a whole layer of neurons*/

template<typename T> struct cacheLayer{
public:
    std::vector<cacheSingle<T>> caches;
};

/*this is a struct of both cache data and activations for the whole
layer. furthermore it is basic type of data generated by forpropagationLayer class*/

template<typename T> struct forpropagationDataLayer{
public:
    std::vector<std::vector<T>> activations;
    cacheLayer<T> caches;
};

/*this class holds functions responsible for forpropagation process*/

template<class T> class forpropagationLayer{
private:

    /*this function calculates vector of logits generated by a single neuron for
    a whole training data set*/

    std::vector<T> calculatingLogit(std::vector<T> weights, std::vector<T> biases, std::vector<std::vector<T>> activations){
        std::vector<T> logit;
        logit = vector::vectorAddition(vector::vectorMatrixElementWiseMultipication(weights, activations), biases);
        return logit;
    }

    /*this function conducts all operations of a single neuron and returns single
    forpropagation data. int mode is code of activation function used in this layer*/

    forpropagationDataSingle<T> singleNeuron(T bias, const std::vector<T>& weights, const std::vector<std::vector<T>>& previousActivations, int mode){
        std::vector<T> biases(weights.size(), bias);
        std::vector<T> logit = calculatingLogit(weights, biases, previousActivations);
        std::vector<T> newActivations = ActivFunctions::start<T>(logit, mode, true);
        cacheSingle<T> localCache = {logit, weights, biases};
        forpropagationDataSingle<T> returnData = {newActivations, localCache};
        return returnData;
    }

public:

    /*this function basicly vectorises singleNeuron() function for a whole layer. 
    int mode is code of activation function used in this layer*/

    forpropagationDataLayer<T> start(const std::vector<std::vector<T>>& weights, const std::vector<T>& biases, const std::vector<std::vector<T>>& activations, int mode){
        forpropagationDataLayer<T> newLayerData;
        std::vector<forpropagationDataSingle<T>> layerActivations;
        std::transform(weights.begin(), weights.end(), std::back_inserter(layerActivations),[this, &biases, &activations, mode, &weights](const std::vector<T>& neuronWeights){return singleNeuron(biases[&neuronWeights - &weights[0]], neuronWeights, activations, mode);});
        newLayerData.activations.resize(layerActivations.size());
        std::transform(layerActivations.begin(), layerActivations.end(), newLayerData.activations.begin(),[](const forpropagationDataSingle<T>& neuronData) { return neuronData.activations; });
        newLayerData.caches.caches = std::move(layerActivations);
        return newLayerData;
    }
};

/*this function connects forpropagationLayer class with every function that
calls it.*/

template<typename T> forpropagationDataLayer<T> startNeuronLayer(const std::vector<std::vector<T>>& weights, const std::vector<T>& biases, const std::vector<std::vector<T>>& activations, int mode){
    forpropagationLayer<T> newLayer;
    return newLayer.start(weights, biases, activations, mode);
}

}; //end of ForPropagaion namespace
#endif //FORWARDPROPAGATION

#ifndef BACKPROPAGATION
#define BACKPROPAGATION

/*this namespace provides functions, classes and data structures
required to implement backward propagation*/

namespace Backpropagation{

/*this is structure of deltas of weights and biases applied
for a whole layer*/

template<typename T> struct deltas{
public:
    std::vector<T> deltaBiases;
    std::vector<T> deltaWeights;
};

/*this class provides all functions needed to implement backpropagation*/

template<class T> class backpropagatingWhatever{
private:

    /*this function calculates deltas of Bias for a whole layer*/

    std::vector<T> deltaBias(std::vector<std::vector<T>> error){
        std::vector<T> result(error.size());
        std::transform(error.begin(), error.end(), result.begin(), [](std::vector<T> lineOfError){return std::accumulate(lineOfError.begin(), lineOfError.end(), T());});
        result = vector::vectorScalarDivision<T>(result, error.size(), true);
        return result;
    }

    /*this function calculates deltas of Weights for a whole layer*/

    std::vector<T> deltaWeights(std::vector<std::vector<T>> error, std::vector<std::vector<T>> activations){
        std::vector<std::vector<T>> firstResult;
        firstResult = vector::matrixElementWiseMultiplication(error, activations);
        std::vector<T> result(firstResult.size());
        std::transform(firstResult.begin(), firstResult.end(), result.begin(), [](std::vector<T> lineOfResult){return std::accumulate(lineOfResult.begin(), lineOfResult.end(), std::vector<T>());});
        result = vector::vectorScalarDivision<T>(result, error.size(), true);
        return result;
    }

    /*this function calculates derivatives of activation functions (vectorized ofc)*/

    std::vector<std::vector<T>> derivativesOfActivations(std::vector<std::vector<T>> logits, int mode){
        std::vector<std::vector<T>> vectorResult(logits.size());
        std::transform(logits.begin(), logits.end(), vectorResult.begin(), [mode](std::vector<T> logitSingleExample){return ActivFunctions::start<T>(logitSingleExample, mode, false);});
        return vectorResult;
    };

    /*this function calculates error of inner layer. mode is 
    activation function for current layer.*/

    std::vector<std::vector<T>> calculateNewError(int mode, std::vector<std::vector<T>> previousError,  std::vector<std::vector<T>> weights,  std::vector<std::vector<T>> logits){
        std::vector<std::vector<T>> derivativesOfActivation = derivativesOfActivations(logits, mode);
        std::vector<std::vector<T>> WtError = vector::matrixElementWiseMultiplication(previousError, weights);
        std::vector<std::vector<T>> newError = vector::matrixElementWiseMultiplication(derivativesOfActivation, WtError);
        return newError;
    };

public:

    /*simply, a public connector to calculateNewError()*/

    std::vector<std::vector<T>> innerErrorLayer(std::vector<std::vector<T>> previousError,  std::vector<std::vector<T>> weights,  std::vector<std::vector<T>> logits, int mode){
        return calculateNewError(mode, previousError, weights, logits);
    };

    /*function to calculate bias and weights deltas*/

    deltas<T> newDeltas(std::vector<std::vector<T>> error, std::vector<std::vector<T>> activations){
        std::vector<T> biases = deltaBias(error);
        std::vector<T> weights = deltaWeights(error, activations);
        deltas<T> newD;
        newD = {biases, weights};
        return newD;
    }
};

/*this function calculates output error, vectorized
only for binary classification!! FUNCTION IS BAD*/

template<typename T> std::vector<std::vector<T>> outputLayerErrorBinaryClassification(std::vector<T> predictedValue, std::vector<T> targetValue){
    std::vector<std::vector<T>> returnValue;
    returnValue.pushback(vector::vectorSubtraction(predictedValue, targetValue));
    return returnValue;
}

/*simply handles everything needed to calculate inner layer error*/

template<typename T> std::vector<std::vector<T>> innerLayerError( std::vector<std::vector<T>> previousError,  std::vector<std::vector<T>> weights,  std::vector<std::vector<T>> logits, int mode){
    std::vector<std::vector<T>> resultError;
    backpropagatingWhatever<T> backPropagation;
    resultError = backPropagation.innerErrorLayer(previousError, weights, logits, mode);
    return resultError;
};

/*this function calculates deltas of biases and weights. 
it doesn't change them!!!*/

template<typename T> deltas<T> calcDelta(std::vector<std::vector<T>> error, std::vector<std::vector<T>> activations){
    std::vector<std::vector<T>> result;
    backpropagatingWhatever<T> newDelta;
    result = newDelta.newDeltas(error, activations);
    return result;
}
}; //end of Backpropagation namespace

#endif //BACKPROPAGATION

#ifndef UPDATINGWANDB
#define UPDATINGWANDB

namespace UpdatingWandB{

template<typename T> struct weightsAndBiases{
public:
    std::vector<std::vector<std::vector<T>>> weights;
    std::vector<std::vector<T>> biases;
};

template<class T> class updatingWandB{
private:
    std::vector<std::vector<T>> newBiases(std::vector<std::vector<T>> deltaBiases, float learningRate, std::vector<std::vector<T>> biases){
        std::vector<std::vector<T>> deltaXrate = vector::vectorMatrixElementWiseMultipication<T>(std::vector<T>{static_cast<T>(learningRate)}, deltaBiases);
        std::vector<std::vector<T>> result = std::transform(biases.begin(), biases.end(), deltaXrate.begin(), result.begin(), 
            [](std::vector<T> a, std::vector<T> b){return vector::vectorSubtraction<T>(a, b);});
        return result;
    }
    std::vector<std::vector<std::vector<T>>> newWeights(std::vector<std::vector<std::vector<T>>> deltaWeights, float learningRate, std::vector<std::vector<std::vector<T>>> weights){
        std::vector<std::vector<std::vector<T>>> deltaXrate = std::transform(deltaWeights.begin(), deltaWeights.end(), deltaXrate.begin(), 
            [learningRate](std::vector<std::vector<T>> array){return vector::vectorMatrixElementWiseMultipication<T>(std::vector<T>{static_cast<T>(learningRate)}, array);});
    }
public:
    std::vector<weightsAndbiases<T>> start(std::vector<std::vector<T>> deltaBias, std::vector<std::vector<std::vector<T>>> deltaWeights, float learningRate, weightsAndBiases<T> data){
        weightsAndBiases<T> result;
        result.biases = newBiases(deltaBias, learningRate, data.biases);
        result.weights = newWeights(deltaWeights, learningRate, data.weights);
        return result;
    }
};

template<typename T> weightsAndBiases<T> updateWeightsAndBiases(std::vector<Backpropagation::deltas<T>>, float learningRate, weightsAndBiases<T> data){
    updatingWandB<T> newWB;
    return newWB.start();
}

};

#endif //UPDATINGWANDB

#ifndef LOSSF
#define lOSSF

namespace LossF{
    
template<class T> class lossF{
private:
    T loss(T predicted, T real){
        T result;
        result = -1 * ((real * log(predicted)) + ((1 - real) * log(1 - predicted)));
        return result;
    }
public:
    T start(std::vector<T> predictedVector, std::vector<T> realVector){
        std::vector<T> everyResult = std::transform(predictedVector.begin(), predictedVector.end(), realVector.begin(), everyResult.begin(), [](T p, T r){return loss(p, r);});
        T result;
        result = std::accumulate(everyResult.begin(), everyResult.end(), 0);
        return result;
    }
};

template<typename T> T calcLoss(std::vector<T> predictedVector, std::vector<T> realVector){
    lossF<T> newLoss;
    T result;
    result = newLoss.start(predictedVector, realVector)
}

};


#endif //LOSSF

#ifndef SINGLELOOP
#define SINGLELOOP

namespace SingleLoop{

template<class T> class loop{
private:
    std::vector<ForPropagation::forpropagationDataLayer<T>> forpropagation(int numberOfLayers, std::vector<std::vector<std::vector<T>>> weights, std::vector<std::vector<T>> biases, std::vector<std::vector<T>> activations, std::vector<int> mode) {
    std::vector<ForPropagation::forpropagationDataLayer<T>> returnData;
    std::vector<std::vector<T>> currentActivations;
    currentActivations = activations;
    for (int i = 0; i < numberOfLayers; i++){
        ForPropagation::forpropagationDataLayer<T> layerData = ForPropagation::startNeuronLayer<T>(weights.at(i), biases.at(i), currentActivations, mode.at(i));
        returnData.push_back(layerData);
        currentActivations = layerData.activations;
    }
    return returnData;
    }
    std::vector<Backpropagation::deltas<T>> backpropagation(int numberOfLayers, std::vector<T> predictedValue, std::vector<T> targetValue, std::vector<std::vector<std::vector<T>>> weights, std::vector<std::vector<T>> biases, std::vector<int> mode, std::vector<std::vector<std::vector<T>>> activations, std::vector<std::vector<std::vector<T>>> logits){
        std::vector<Backpropagation::deltas<T>> deltas;
        std::vector<std::vector<std::vector<T>>> errors;
        errors.push_back(Backpropagation::outputLayerErrorBinaryClassification<T>(predictedValue, targetValue));
        for (int i = 0; i < (numberOfLayers - 1); i++){
            errors.push_back(Backpropagation::innerLayerError<T>(errors.at(i - 1), weights.at(numberOfLayers - i - 1), logits.at(numberOfLayers - i - 1), mode.at(numberOfLayers - i - 1)));
        }
        for(int i = 0; i < numberOfLayers; i++){
            deltas.push_back(Backpropagation::calcDelta<T>(errors.at(numberOfLayers - i - 1), activations.at(i)));
        }
        return deltas;
    }
    UpdatingWandB::weightsAndBiases<T> updatingWandB(std::vector<std::vector<T>> deltaBias, std::vector<std::vector<std::vector<T>>> deltaWeights, float learningRate, weightsAndBiases<T> data){
        UpdatingWandB::weightsAndBiases<T> newWandB;
        newWandB = UpdatingWandB::updateWeightsAndBiases<T>(deltas, learningRate, data);
        return newWandB;
    }
    T loss(std::vector<T> predicted, std::vector<T> real){
        T loss = LossF::calcLoss<T>(predicted, real);
        return loss;
    }
public:
    void mainLoop(std::vector<int> config, std::vector<std::vector<std::vector<T>>> weights, std::vector<std::vector<T>> biases, std::vector<T> activations, std::vector<T> target, ){
        std::vector<ForPropagation::forpropagationDataLayer<T>> cache;
        cache = forpropagation(config.at(1), weights, biases, activations, vector::vectorSlicer<int>(config, (config.at(1) + 1), (config.at(i) * 2 + 1)));
        std::vector<Backpropagation::deltas<T>> deltas;
        std::vector<std::vector<std::vector<T>>> activationsNew;
        std::transform(cache.begin(), cache.end(), activationsNew.begin(), []{ForPropagation::forpropagationLayer<T> layer}(return layer.activations;));
        std::vector<ForPropagation::cacheLayer<T>> logitsNewLayer;
        std::transform(cache.begin(), cache.end(), logitsNewLayer.begin(), []{ForPropagation::fcacheLayer<T> layer}(return layer.caches;));
        std::vector<ForPropagation::cacheSingle<T>> logitsNew;
        std::transform(logitsNewLayer.begin(), logitsNewLayer.end(), logitsNew.begin(), []{ForPropagation::cacheSingle<T> layer}(return layer.caches;));
        std::vector<std::vector<T>> logitsFinal;
        std::transform(logitsNew.begin(), logitsNew.end(), logitsFinal.begin(), []{ForPropagation::cacheSingle<T> cache}(return cache.logit;));
        deltas = backpropagation(config.at(1), cache.at(config.at(1)).activations, target, weights, biases, vector::vectorSlicer<int>(config, (config.at(1) + 1), (config.at(i) * 2 + 1)), activationsNew, logitsFinal);
        loss();
        updatingWandB();
    };
};

};

#endif //SINGLELOOP

int main(int argc, char const *argv[]){
    if(argc == 2){
        startmenu::starter("0");
    }else{
        startmenu::starter("1", {argv[2], argv[3], argv[4], argv[5]});
    }
    
    return 0;
}
//EOF
//MAM DOŚĆ TEJ KURWY