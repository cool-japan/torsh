function result = torsh_add(A, B)
    % TORSH_ADD Element-wise addition of two ToRSh tensors
    %
    % RESULT = TORSH_ADD(A, B) performs element-wise addition of tensors A and B.
    %
    % Examples:
    %   A = torsh_tensor([1, 2; 3, 4]);
    %   B = torsh_tensor([5, 6; 7, 8]);
    %   C = torsh_add(A, B);  % Results in [6, 8; 10, 12]
    
    if nargin < 2
        error('torsh_add:MissingInputs', 'Two input tensors are required');
    end
    
    % Call the MEX function for tensor addition
    result = torsh_mex('add', A, B);
end