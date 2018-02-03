function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X]; %X is now a 5000*401 matrix
A1=X;               %doing it for naming simplicity

z1=A1*(Theta1)';    %A1 is 5000*401 (Theta1)' is 401*25 hence resultant 5000*25

A2=sigmoid(z1);     %every layer has logistic regression hypotheis hence sigmoid of (Theta_T*X)   
A2=[ones(m,1) A2];  %Adding bias unit for computational purposes hence A2 is 401*26 matrix

z2=A2*(Theta2)';  %resultant is 5000*10 matrix  

A3=sigmoid(z2); %sigmoid of A2*Theta2 results in A3 i,e 3rd layer values

[m,i]=max(A3,[],2);  % we have P has 5000*1 matrix, our answer now is 5000*10 matrix, hence using max-
                     %-function. The variable i will contain the index of
                     %maximum value of each row i,e between 1 to 10 and is
                     %a column vector with m*1 dimension
p=i;                 


% =========================================================================


end
