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
        function obj = NeuralNet(layers,layerDimensions)
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.weights = cell(layers);
            for i=1:length(obj.weights)
                obj.weights{i} = cell(layerDimensions(i));
                %need to build d3
            end
            
            obj.biases = cell(layers);
            for i=1:length(obj.biases)
                obj.biases{i} = cell(layerDimensions(i));
            end
            
        end
        
%         function outputArg = method1(obj,inputArg)
%             %METHOD1 Summary of this method goes here
%             %   Detailed explanation goes here
%             outputArg = obj.Property1 + inputArg;
%         end

    end
end

        
