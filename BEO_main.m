function BEO_main()
%------------------Bayesian Evolutionary Optimization----------------------
clear
clc
addpath(genpath(pwd));
%Parameter setting
fucnum = 1; %Ellipsoid function
D = 10;
FEmax = 11*D;
gmax = 20;
theta = 5.*ones(1,D);

%% Initialization
lu = Initialize(fucnum,D);
Lb = lu(1,:);
Ub = lu(2,:);
Boundary = [Ub;Lb];
%Initialize the training archive
Nt = 2*D;
Alhs = lhsdesign(Nt,D);
Arc = repmat(Lb,Nt,1) + Alhs.* repmat((Ub-Lb),Nt,1);
Arc = unique(Arc,'rows');
Arc(:,D+1) = fitness(Arc, fucnum);
FEnum = size(Arc,1);
%Initialize a parent population
Np = 50;
Plhs = lhsdesign(Np,D);
Parent = repmat(Lb,Np,1) + Plhs.* repmat((Ub-Lb),Np,1);
%Optimization
while FEnum < FEmax
    ArcDec = Arc(:,1:D);
    ArcObj = Arc(:,end);
    %Train a GP model
    GPmodel = dacefit(ArcDec,ArcObj,'regpoly0','corrgauss',theta,1e-5.*ones(1,D),100.*ones(1,D));
    %Generate an offspring
    g = 0;
    while g < gmax
        g = g+1;
        Offspring = GA(Parent,Boundary);
        Popcom = [Parent;Offspring];
        N = size(Popcom,1);
        for i = 1: N
            [PopObj(i,1),~,MSE(i,1)] = predictor(Popcom(i,:),GPmodel);
        end
        [Popsort,index] = sort(PopObj,'ascend');
        Parent = Popcom(index(1:Np),:);
        MSEp = MSE(index(1:Np),:);
        ParObj = PopObj(index(1:Np));
    end
    %Infill criterion based on EI
    Abest = min(Arc(:,end));
    s = sqrt(MSEp);
    lamda = (repmat(Abest,Np,1)-ParObj)./s;
    for i = 1:Np
        EI(i,1) = (Abest-ParObj(i)).* Gaussian_CDF(lamda(i)) + s(i)*Gaussian_PDF(lamda(i));
    end
    [~,index] = max(EI);
    Popreal = Parent(index,:);
    Popreal(:,D+1) = fitness(Popreal, fucnum);
    FEnum = FEnum + size(Popreal,1)
    %Update the training archive
    Arc = [Arc; Popreal];
    Arc = unique(Arc,'rows');
    Pbest = min(Arc(:,end))
    plot(FEnum,Pbest,'-o');
    hold on
    getframe;
end
return