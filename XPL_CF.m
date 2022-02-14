function [A, B, D, a, b, S, Z, costValue, RMSEValid, bestIterNum] = ...
          XPL_CF(X_data, R, muA, muB, lambda, eta, ...
          ADMMiter, maxIter, maskTrain, maskValid)
%%
% This code solves the XPL-CF model in the following paper:
% 
% Almutairi, F. M., Sidiropoulos, N. D. , and Yang, B. "XPL-CF: Explainable 
% Embeddings for Feature-based Collaborative Filtering." Proceedings of the 
% 30th ACM CIKM 2021.
% 
% 
% 
% The code solves the following optimization problem:
% (Eq.5 in the paper with the same notation)
% 
%       min_{A,B,d,a,b,S,Z}    0.5 * ||W .* (X - A * D * B^T - a * 1^T - 1 * b^T)||^2 
%               + lambda/2 * 1^T * (S + Z^T) * 1 
%               + muA/2 * ||A - S * B||^2 + muB/2 * ||B - Z * A||^2
%               + eta (||a||^2 + ||b||^2 + ||d||^2)
%       s.t.   S, Z^T >= 0; ||A(:,r)||_2  = ||B(:,r)||_2  = 1; d = Diag(d)
% 
% 
% The inputs of this function are as follows:
%       X_data: the (user x item) rating matrix of size NxM, with missing 
%               entries defined as NaN
%       R: is the matrix rank (hyper-parameter)
%       muA, muB, lambda, eta: are regularization hyper-parameters
%       ADMMiter: is the maximum number of iterations for the ADMM inner
%                 loops to solve the variables A, B, S, and Z
%       maxIter: is the maximum number of iteration of the main algorithm
%       maskTrain: is an NxM LOGICAL matrix with ones at the TRAIN data indices,
%                  and zeros otherwise.
%       maskValid: is an NxM LOGICAL matrix with ones at the VALIDATION data indices,
%                  and zeros otherwise.
% 
% 
% 
%  The outputs of this function are as follows:
%       A, B, D, a, b, S, Z: are the variables as defined in Eq.5
%       costValue: is the value of the cost function at every iteration   
%       RMSEValid: is the Root Mean Square Error of the VLAIDATION data 
%       bestIterNum: is the iteration number before RMSEValid starts
%                    increasing (the model overfits) 
% 
% 
% Faisal Almutairi (almut012@umn.edu), May 2021


%% ================ define and initialize variables =======================
% define the training data
[N, M] = size(X_data);
X0 = zeros(N, M);
X0(maskTrain) = X_data(maskTrain);

% initialize variables A, B, D, S, Z (and some ADMM intermediate variables)
A = rand(N,R); A = A/diag(sqrt(sum(A.^2)));
B = rand(M,R); B = B/diag(sqrt(sum(B.^2)));
D = eye(R);
a = zeros(N, 1);
b = zeros(M, 1);

A_old = A;
A_tilde = A';
Ua = zeros(N, R);
FLAG_a = zeros(1, N);

B_old = B;
B_tilde = B';
Ub = zeros(M, R);
FLAG_b = zeros(1, M);

S = rand(N, M); 
S_old = S;
UsA = zeros(N, M);

Z = rand(M, N); 
Z_old = Z;
UsB = zeros(M, N);

% define some useful variables
admm_tol = 1e-3;
cost_tol = 1e-07;
costValue = zeros(1, maxIter);
RMSE_old = Inf;
bestIterNum = 0;
cost_old = 0;
diff = Inf;
outeriter = 0;

%% ======================= main algorithm ================================= 
while (outeriter < maxIter) && (diff > cost_tol)
    outeriter = outeriter + 1;
%     disp(outeriter)

    X_tilde = X0 - (a * ones(1, M)) - (ones(N, 1) * b');
    X_tildeU = cell(N, 1);
    for i = 1:N
        X_tildeU{i} = X_tilde(i,maskTrain(i,:));
    end
    X_tildeI = cell(1, M);
    for j = 1:M
        X_tildeI{j} = X_tilde(maskTrain(:, j), j);
    end
    %% update A: ADMNM algorithm
    BD = B*D;  
    rho = norm(BD,'fro')^2/(R); 
    BSa = sqrt(muA) * (S * B)';
        
    for iter = 1:ADMMiter  
        parfor i = 1:N
            Wa = [BD(maskTrain(i,:),:); sqrt(muA) * eye(R)];                 
            [La, FLAG_a(i)] = chol(Wa'* Wa + rho*eye(R), 'lower');
            Ya = [X_tildeU{i}'; BSa(:, i)];            
            Fa = Wa'*Ya;
            A_tilde(:,i) = La' \ (La \ (Fa + rho*(A(i,:) + Ua(i,:))'));
        end             
        if any(FLAG_a)
            disp('matrix is not positive definite to solve A, change hyper-parameters')
            RMSEValid = Inf;
            return
        end        
        parfor r = 1:R
            Ar = A_tilde(r,:)' - Ua(:,r);
            A(:,r) = Ar/norm(Ar);
        end
        Ua = Ua + A - A_tilde'; 
        
        pA = norm(A - A_tilde','fro')^2/norm(A,'fro')^2;       
        dA = norm(A - A_old,'fro')^2/norm(Ua,'fro')^2;
        A_old = A;
        if pA < admm_tol && dA < admm_tol
            break;
        end      
    end            
    %% update B: ADMM algorithm 
    AD = A*D;  
    rho = norm(AD,'fro')^2/(R); 
    ASb = sqrt(muB) * (Z * A)';
        
    for iter = 1:ADMMiter  
        parfor j = 1:M
            Wb = [AD(maskTrain(:,j),:); sqrt(muB) * eye(R)];                 
            [Lb, FLAG_b(j)] = chol(Wb'*Wb + rho*eye(R), 'lower');
            Yb = [X_tildeI{j}; ASb(:, j)];            
            Fb = Wb'*Yb;
            B_tilde(:,j) = Lb' \ (Lb \ (Fb + rho*(B(j,:) + Ub(j,:))'));
        end             
        if any(FLAG_b)
            disp('matrix is not positive definite to solve B, change hyper-parameters')
            RMSEValid = Inf;
            return
        end        
        parfor r = 1:R
            Br = B_tilde(r,:)' - Ub(:,r);
            B(:,r) = Br/norm(Br);
        end
        Ub = Ub + B - B_tilde'; 
        
        pB = norm(B - B_tilde','fro')^2/norm(B,'fro')^2;       
        dB = norm(B - B_old,'fro')^2/norm(Ub,'fro')^2;
        B_old = B;
        if pB < admm_tol && dB < admm_tol
            break;
        end      
    end   
    %% update D: Closed-form   
    x = [X_tilde(maskTrain(:)); zeros(R,1)];   
    BkrA = kr(B,A);
    BkrA = [BkrA(maskTrain(:),:); sqrt(eta) * eye(R)];
    d = BkrA\x;
    D = diag(d);
    %% Update users' biases
    for i=1:N
        num = sum(maskTrain(i,:));
        Xi = X0(i,maskTrain(i,:))' - b(maskTrain(i,:)) - B(maskTrain(i,:),:) * D * A(i,:)';
        yy = [Xi; 0];
        XX = [ones(num,1); sqrt(eta)];
        a(i) = XX\yy;
    end
    %% Update items' biases
    for i = 1:M
        num = sum(maskTrain(:,i));
        Xi = X0(maskTrain(:,i),i) - a(maskTrain(:,i)) - A(maskTrain(:,i),:) * D * B(i,:)';
        yy = [Xi; 0];
        XX = [ones(num,1); sqrt(eta)];
        b(i) = XX\yy;
    end
    %% update S: ADMM algorithm        
    rho = norm(B,'fro')^2/(M);
    BB = B*B';
    [LsA, FLAG_sA] = chol(muA * BB + rho * eye(M), 'lower');
    if FLAG_sA ~= 0
        disp('matrix is not positive definite to solve S, change hyper-parameters')
        RMSEValid = Inf;
        break
    end    
    FsA = muA*B*A' - lambda*ones(M,N); 
    for iter = 1:ADMMiter
        Sa_tilde = LsA' \ (LsA \ ( FsA + rho * (S + UsA)'));
        S = max(0, Sa_tilde' - UsA);
        UsA = UsA + S - Sa_tilde';
        
        pSa = norm(S - Sa_tilde','fro')^2/norm(S,'fro')^2;
        dSa = norm(S - S_old,'fro')^2/norm(UsA,'fro')^2;            
        S_old = S;
        if pSa < admm_tol && dSa < admm_tol
            break;
        end 
    end
    
    if isempty(find(S, 1))
        disp('lambda is too large (all-zeros S)')
        RMSEValid = Inf;
        break
    end
    %% update Z: ADMM algorithm        
    rho = norm(A,'fro')^2/(N);
    AA = A*A';
    [LsB, FLAG_sB] = chol(muB * AA + rho * eye(N), 'lower');
    if FLAG_sB ~= 0
        disp('matrix is not positive definite to solve Z, change hyper-parameters')
        RMSEValid = Inf;
        break
    end    
    FsB = muB*A*B' - lambda*ones(N, M); 
    for iter = 1:ADMMiter
        Sb_tilde = LsB' \ (LsB \ ( FsB + rho * (Z + UsB)'));
        Z = max(0, Sb_tilde' - UsB);
        UsB = UsB + Z - Sb_tilde';
        
        pSb = norm(Z - Sb_tilde','fro')^2/norm(Z,'fro')^2;
        dSb = norm(Z - Z_old,'fro')^2/norm(UsB,'fro')^2;            
        Z_old = Z;
        if pSb < admm_tol && dSb < admm_tol
            break;
        end 
    end
    
    if isempty(find(Z, 1))
        disp('lambda is too large (all-zeros Z)')
        RMSEValid = Inf;
        break
    end 
    %% calculate cost value
    X_hat = A*D*B' + a*ones(1,M) + ones(N,1)*b';
    costValue(outeriter) = 0.5 * norm(maskTrain .* (X0 - X_hat), 'fro')^2 ...
        + muA/2 * norm(A - S * B, 'fro')^2 + muB/2 * norm(B - Z * A, 'fro')^2 ...
        + lambda * sum(S(:)) + lambda * sum(Z(:)) + eta/2 * (norm(d)^2 + norm(a)^2 + norm(b)^2);
    
    diff = abs(costValue(outeriter) - cost_old);
    cost_old = costValue(outeriter);
    if isnan(diff)
        RMSEValid = Inf;
        disp('XPL-CF fails for these hyper-parameters')
    end
    %% calculate RMSE on validation data
    if (nargin > 9) && (mod(outeriter,5) == 0)
        x_min = 0; %min(min(X_train(mask_train)));
        x_max = Inf; %max(max(X_train(mask_train)));
        RMSEValid = RMSE(X_data(maskValid), (X_hat(maskValid)), x_min, x_max);
         disp(['*****XPL-CF: Validation Data RMSE: ', num2str(RMSEValid), '*****'])
        % 0.0005  is a safe margin
        if RMSEValid > RMSE_old + 0.0005  
            disp('Stopped cause RMSE degraded')
            RMSEValid = RMSE_old; 
            bestIterNum = outeriter - 5;
            break        
        end
        RMSE_old = RMSEValid;  
    end 
    bestIterNum = outeriter; 
end

end