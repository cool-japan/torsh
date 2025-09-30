function result = torsh_matmul(A, B)
    % TORSH_MATMUL Matrix multiplication of two ToRSh tensors
    %
    % RESULT = TORSH_MATMUL(A, B) performs matrix multiplication of tensors A and B.
    %
    % Examples:
    %   A = torsh_tensor([1, 2; 3, 4]);
    %   B = torsh_tensor([5, 6; 7, 8]);
    %   C = torsh_matmul(A, B);  % Matrix multiplication result
    
    if nargin < 2
        error('torsh_matmul:MissingInputs', 'Two input tensors are required');
    end
    
    % Call the MEX function for matrix multiplication
    result = torsh_mex('matmul', A, B);
end