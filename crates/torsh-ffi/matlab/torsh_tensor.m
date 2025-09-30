function tensor = torsh_tensor(data)
    % TORSH_TENSOR Create a ToRSh tensor from MATLAB data
    %
    % TENSOR = TORSH_TENSOR(DATA) creates a ToRSh tensor from the input DATA.
    % DATA can be a scalar, vector, matrix, or N-dimensional array.
    %
    % Examples:
    %   % Create a 2x3 tensor
    %   A = torsh_tensor([1, 2, 3; 4, 5, 6]);
    %
    %   % Create a tensor from random data
    %   B = torsh_tensor(randn(10, 20));
    %
    %   % Create a scalar tensor
    %   C = torsh_tensor(42);
    
    if nargin < 1
        error('torsh_tensor:MissingInput', 'Input data is required');
    end
    
    % Convert input to double if necessary
    if ~isa(data, 'double')
        data = double(data);
    end
    
    % Call the MEX function (this would be compiled from our Rust code)
    tensor = torsh_mex('create_tensor', data);
end