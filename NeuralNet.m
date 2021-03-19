classdef NeuralNet
    %Basic Neural Net
    %   Neural net seperated into multiple layers
    
    properties
        weights = {{[]}};
            %3D array of weights
            %d1 Layer number
            %d2 Neuron number
            %d3 Weight corresponding to neurons in previous layer
        biases = {{}};
            %one bias for each neuron fn
        
    end
    
    methods
        function obj = NeuralNet(layers,layerDimension)
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.weights = cell(1, layers);
            for i=1:length(obj.weights)
                obj.weights{i} = cell(layerDimension, 1);
                for j=1:layerDimension
                    %Random number from -1 to 1
                    obj.weights{i}{j} = 2.*rand(1, layerDimension) -1;
                end
            end
            
            obj.biases = cell(1, layers);
            for i=1:length(obj.biases)
                obj.biases{i} = cell(layerDimension, 1);
            end

        end
        function sig = sigmoid(
%         
%         function runNet
%         end
%         function calculateError
%         end
%         function outputArg = method1(obj,inputArg)
%             %METHOD1 Summary of this method goes here
%             %   Detailed explanation goes here
%             outputArg = obj.Property1 + inputArg;
%         end
    end
end

