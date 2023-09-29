clc;
clear;
close all;

%% Problem Definition
load('Indian_pines_corrected.mat')
load('Indian_pines.mat')
load('Indian_pines_gt.mat')

y= Convert2d(indian_pines_gt(:,:,1));


[hh,ww,nBand]= size(indian_pines);
nVar=21025;                         % Number of Variables

VarSize=[1 nVar];            % Size of Variables Matrix

VarMin=0;                    % Lower Bound of Variables
VarMax= 16;                  % Upper Bound of Variables

VarRange=[VarMin VarMax];    % Variation Range of Variables

VelMax=(VarMax-VarMin)/10;   % Maximum Velocity
VelMin=-VelMax;              % Minimum Velocity

%% PSO Parameters

MaxIt=1000;         % Maximum Number of Iterations
nPop=50;            % Swarm (Population) Size

% Definition of Constriction Coefficients
phi1=2.05;
phi2=2.05;
phi=phi1+phi2;
chi=2/(phi-2+sqrt(phi^2-4*phi));

w=chi;
c1=phi1*chi;
c2=phi2*chi;

%% Initialization

% Empty Structure to Hold Individuals Data
empty_individual.Position=[];
empty_individual.Velocity=[];
empty_individual.Cost=[];
empty_individual.Best.Position=[];
empty_individual.Best.Cost=[];

% Create Population Matrix
pop=repmat(empty_individual,nPop,1);

% Global Best
BestSol.Cost=inf;
Indx=randi(nBand,[1,nPop]);
% Initialize Positions
for i=1:nPop
    
    x1=Convert2d(indian_pines(:,:,Indx(i)));
   pop(i).Position=16*(x1-min(x1))./(max(x1)-min(x1)); % normalized x1, 0 <=x1<=16
    pop(i).Velocity=zeros(VarSize);
    pop(i).Cost=CostFunction(pop(i).Position,y);
  
    
    pop(i).Best.Position=pop(i).Position;
    pop(i).Best.Cost=pop(i).Cost;
    
    if pop(i).Best.Cost<BestSol.Cost
        BestSol=pop(i).Best;
    end
    
end

% Vector to Hold Best Cost Values
BestCost=zeros(MaxIt,1);


%% PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        pop(i).Velocity=w*pop(i).Velocity ...
            + c1*rand(VarSize).*(pop(i).Best.Position-pop(i).Position) ...
            + c2*rand(VarSize).*(BestSol.Position-pop(i).Position);
        
        % Apply Velocity Bounds
        pop(i).Velocity=min(max(pop(i).Velocity,VelMin),VelMax);
        
        % Update Position
        pop(i).Position=pop(i).Position+pop(i).Velocity;
        
        % Velocity Reflection
        flag=(pop(i).Position<VarMin | pop(i).Position>VarMax);
        pop(i).Velocity(flag)=-pop(i).Velocity(flag);
        
        % Apply Position Bounds
        pop(i).Position=min(max(pop(i).Position,VarMin),VarMax);
        
        % Evaluation
         x1=Convert2d(indian_pines(:,:,Indx(i)));
        pop(i).Position=16*(x1-min(x1))./(max(x1)-min(x1)); % normalized x1, 0 <=x1<=16
        pop(i).Cost=CostFunction(pop(i).Position,y);
        
        % Update Personal Best
        if pop(i).Cost<pop(i).Best.Cost
            
            pop(i).Best.Position=pop(i).Position;
            pop(i).Best.Cost=pop(i).Cost;
            
            % Update Global Best
            if pop(i).Best.Cost<BestSol.Cost
                BestSol=pop(i).Best;
            end
            
        end
        
    end
    
    % Store Best Cost
    BestCost(it)=BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end

%% Plots

figure;
semilogy(BestCost);

%convert from 2D to 3D
ClassifiedMap3D = Convert3d(pop(1).Position,hh,ww,nBand);
figure, imagesc(ClassifiedMap3D),title('Output PSO Classified Map'),axis off

