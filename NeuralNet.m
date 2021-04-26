classdef NeuralNet
    %Basic Neural Net
    %   Neural net seperated into multiple layers
    
    properties
        weights;
            %3D array of weights
            %d1 Layer number
            %d2 Neuron number
            %d3 Weight corresponding to neurons in previous layer
        layerLen;
            %how long each layer is (must be same for all layers)
        numLayers;
            %how many layers there are
        
    end
    
    methods
        function obj = NeuralNet(layers,layerDimension)
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.numLayers = layers;
            obj.layerLen = layerDimension;
            obj.weights = ones(layers, layerDimension, layerDimension);
            
            for i = 1:layers
                for j = 1:layerDimension
                    obj.weights(i, j, :) = 2.*rand(1, 1, layerDimension) -1;%fix this
                end
            end

            obj.weights

        end
        

         function out = runNet(obj, in)
             currLayer = in;
             newLayer = in;
             for i = 1:obj.numLayers
                 for j = 1:obj.layerLen
                    newLayer(j) = sigmoid(dot(currLayer, squeeze(obj.weights(i, j, :))));
                 end
                 currLayer = newLayer;
             end
             
             out = currLayer;
         end
         
         function accuracy = trainNet(obj, trainingIns, trainingOuts, iterations)
             %training ins/outs are 2d arrays of system inputs/outputs with
             %each column identifying a new case
             for iter = 1:iterations
                %Training iterations
                for trainingCase = 1:size(trainingIns, 1)
                    layers = ones(obj.numLayers, obj.layerLen);
                    currIn = trainingIns(trainingCase);
                    currOut = trainingOuts(trainingCase);

                    for i = 1:obj.numLayers
                        for j = 1:obj.layerLen
                            layers(i, j) = sigmoid(dot(currIn, squeeze(obj.weights(i, j, :))));
                        end
                        currIn = squeeze(layers(i, :));
                    end
                    response = currIn;
                    
                    
                    %CALCULATE ERRORS
                    finalError = response - currOut%error in outputs
                    finalDelta = finalError *sigD(squeeze(layers(obj.numLayers, :)));%finds error in last layer
                  
                    %for the rest of the layers
                    % error of previous layer =  delta dot wieghts
                    % previous delta = previous error * sigd(prev weights)
                    
                    %UPDATE WEIGHTS
                    %for each set of weights
                    %weights -= dot(startlayer, finlayerdelta)
                end
             end
         end
         %add something to make the outputs the same length as
                    %inputs/layers
    end
    methods(Static)
         function sig = sigmoidFn(x)
            sig = 1/(1+exp(-x));
         end
         function sigD = sigD(x)
             sigD = x .*(1-x);
         end
    end
end

