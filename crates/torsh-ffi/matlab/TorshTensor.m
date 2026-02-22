classdef TorshTensor < handle
    % TORSHTENSOR MATLAB class wrapper for ToRSh tensors
    %
    % This class provides a MATLAB-friendly interface to ToRSh tensor operations.
    % It uses MEX functions to interface with the Rust-based ToRSh library.
    %
    % Examples:
    %   % Create tensors
    %   A = TorshTensor([1, 2; 3, 4]);
    %   B = TorshTensor([5, 6; 7, 8]);
    %
    %   % Perform operations
    %   C = A + B;        % Element-wise addition
    %   D = A * B;        % Matrix multiplication
    %   E = A.relu();     % ReLU activation
    %
    %   % Access data
    %   data = A.data();  % Get MATLAB array
    
    properties (Access = private)
        tensor_handle  % Handle to the underlying ToRSh tensor
    end
    
    methods
        function obj = TorshTensor(data)
            % Constructor: Create TorshTensor from MATLAB data
            if nargin > 0
                if isa(data, 'TorshTensor')
                    % Copy constructor
                    obj.tensor_handle = torsh_mex('copy', data.tensor_handle);
                else
                    % Create from data
                    obj.tensor_handle = torsh_mex('create', double(data));
                end
            end
        end
        
        function delete(obj)
            % Destructor: Clean up the tensor handle
            if ~isempty(obj.tensor_handle)
                torsh_mex('free', obj.tensor_handle);
            end
        end
        
        function data = data(obj)
            % Get the tensor data as a MATLAB array
            data = torsh_mex('data', obj.tensor_handle);
        end
        
        function dims = size(obj, varargin)
            % Get tensor dimensions
            if nargin == 1
                dims = torsh_mex('size', obj.tensor_handle);
            else
                all_dims = torsh_mex('size', obj.tensor_handle);
                dims = all_dims(varargin{1});
            end
        end
        
        function n = numel(obj)
            % Get total number of elements
            n = torsh_mex('numel', obj.tensor_handle);
        end
        
        function n = ndims(obj)
            % Get number of dimensions
            n = torsh_mex('ndims', obj.tensor_handle);
        end
        
        % Arithmetic operations
        function result = plus(obj, other)
            % Element-wise addition
            if isa(other, 'TorshTensor')
                result_handle = torsh_mex('add', obj.tensor_handle, other.tensor_handle);
            else
                % Scalar addition
                result_handle = torsh_mex('add_scalar', obj.tensor_handle, double(other));
            end
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = minus(obj, other)
            % Element-wise subtraction
            if isa(other, 'TorshTensor')
                result_handle = torsh_mex('sub', obj.tensor_handle, other.tensor_handle);
            else
                % Scalar subtraction
                result_handle = torsh_mex('sub_scalar', obj.tensor_handle, double(other));
            end
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = times(obj, other)
            % Element-wise multiplication
            if isa(other, 'TorshTensor')
                result_handle = torsh_mex('mul', obj.tensor_handle, other.tensor_handle);
            else
                % Scalar multiplication
                result_handle = torsh_mex('mul_scalar', obj.tensor_handle, double(other));
            end
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = mtimes(obj, other)
            % Matrix multiplication
            if isa(other, 'TorshTensor')
                result_handle = torsh_mex('matmul', obj.tensor_handle, other.tensor_handle);
            else
                % Scalar multiplication (same as times for scalars)
                result_handle = torsh_mex('mul_scalar', obj.tensor_handle, double(other));
            end
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = rdivide(obj, other)
            % Element-wise division
            if isa(other, 'TorshTensor')
                result_handle = torsh_mex('div', obj.tensor_handle, other.tensor_handle);
            else
                % Scalar division
                result_handle = torsh_mex('div_scalar', obj.tensor_handle, double(other));
            end
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        % Neural network operations
        function result = relu(obj)
            % ReLU activation function
            result_handle = torsh_mex('relu', obj.tensor_handle);
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = sigmoid(obj)
            % Sigmoid activation function
            result_handle = torsh_mex('sigmoid', obj.tensor_handle);
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = tanh(obj)
            % Tanh activation function
            result_handle = torsh_mex('tanh', obj.tensor_handle);
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = softmax(obj, dim)
            % Softmax activation function
            if nargin < 2
                dim = -1;  % Last dimension by default
            end
            result_handle = torsh_mex('softmax', obj.tensor_handle, int32(dim));
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        % Reduction operations
        function result = sum(obj, dim, keepdim)
            % Sum along specified dimension
            if nargin < 2
                % Sum all elements
                result_handle = torsh_mex('sum_all', obj.tensor_handle);
            else
                if nargin < 3
                    keepdim = false;
                end
                result_handle = torsh_mex('sum_dim', obj.tensor_handle, int32(dim), logical(keepdim));
            end
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = mean(obj, dim, keepdim)
            % Mean along specified dimension
            if nargin < 2
                result_handle = torsh_mex('mean_all', obj.tensor_handle);
            else
                if nargin < 3
                    keepdim = false;
                end
                result_handle = torsh_mex('mean_dim', obj.tensor_handle, int32(dim), logical(keepdim));
            end
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        % Reshape and view operations
        function result = reshape(obj, varargin)
            % Reshape tensor to new dimensions
            new_shape = [varargin{:}];
            result_handle = torsh_mex('reshape', obj.tensor_handle, int32(new_shape));
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = transpose(obj, dim1, dim2)
            % Transpose two dimensions
            if nargin < 3
                % Default transpose for 2D tensors
                dim1 = 0;
                dim2 = 1;
            end
            result_handle = torsh_mex('transpose', obj.tensor_handle, int32(dim1), int32(dim2));
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        % Display methods
        function disp(obj)
            % Display tensor information
            if isempty(obj.tensor_handle)
                fprintf('Empty TorshTensor\n');
                return;
            end
            
            dims = obj.size();
            fprintf('TorshTensor of size ');
            fprintf('%d', dims(1));
            for i = 2:length(dims)
                fprintf('x%d', dims(i));
            end
            fprintf('\n');
            
            % Display data for small tensors
            if obj.numel() <= 100
                data = obj.data();
                disp(data);
            else
                fprintf('(data not displayed for large tensors)\n');
            end
        end
    end
    
    methods (Static)
        function result = zeros(varargin)
            % Create tensor filled with zeros
            dims = [varargin{:}];
            result_handle = torsh_mex('zeros', int32(dims));
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = ones(varargin)
            % Create tensor filled with ones
            dims = [varargin{:}];
            result_handle = torsh_mex('ones', int32(dims));
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = randn(varargin)
            % Create tensor with random normal values
            dims = [varargin{:}];
            result_handle = torsh_mex('randn', int32(dims));
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
        
        function result = eye(n, m)
            % Create identity matrix
            if nargin < 2
                m = n;
            end
            result_handle = torsh_mex('eye', int32(n), int32(m));
            result = TorshTensor();
            result.tensor_handle = result_handle;
        end
    end
end