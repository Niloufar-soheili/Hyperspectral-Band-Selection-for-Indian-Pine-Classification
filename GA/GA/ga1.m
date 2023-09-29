clc;
clear;
close all;

load('Indian_pines_corrected.mat')
load('Indian_pines.mat')
load('Indian_pines_gt.mat')


%% Problem Definition

y= Convert2d(indian_pines_gt(:,:,1));

[h,w,nBand]= size(indian_pines);


nVar=21025;                    % Number of Variables

VarSize=[1 nVar];           % Size of Variables Matrix

VarMin=0;                    % Lower Bound of Variables
VarMax= 16;                  % Upper Bound of Variables

VarRange=[VarMin VarMax];   % Variation Range of Variables

%% GA Parameters

MaxIt=1000;      % Maximum Number of Iterations

nPop=50;         % Population Size

pCrossover=0.7;                         % Crossover Percentage
nCrossover=round(pCrossover*nPop/2)*2;  % Number of Parents (Offsprings)

pMutation=0.2;                      % Mutation Percentage
nMutation=round(pMutation*nPop);    % Number of Mutants


%% Initialization

% Empty Structure to Hold Individuals Data
empty_individual.Position=[];
empty_individual.Cost=[];

% Create Population Matrix
pop=repmat(empty_individual,nPop,1);
Indx=randi(nBand,[1,nPop]);

% Initialize Positions
for i=1:nPop
    x1=Convert2d(indian_pines(:,:,Indx(i)));
   pop(i).Position=16*(x1-min(x1))./(max(x1)-min(x1)); % normalized x1, 0 <=x1<=16
   % pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).Cost=CostFunction(pop(i).Position,y);
end

% Sort Population
[pop SortOrder]=SortPopulation(pop);

% Store Best Solution
BestSol=pop(1);

% Vector to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%% GA Main Loop

for it=1:MaxIt
    
    % Crossover
    popc=repmat(empty_individual,nCrossover/2,2);
    for k=1:nCrossover/2
        
        i1=randi([1 nPop]);
        i2=randi([1 nPop]);
        
        p1=pop(i1);
        p2=pop(i2);
        
        [popc(k,1).Position popc(k,2).Position]=Crossover(p1.Position,p2.Position,VarRange);
        
        popc(k,1).Cost=CostFunction(popc(k,1).Position,y);
        popc(k,2).Cost=CostFunction(popc(k,2).Position,y);
        
    end
    popc=popc(:);
    
    
    % Mutation
    popm=repmat(empty_individual,nMutation,1);
    for k=1:nMutation
        
        i=randi([1 nPop]);
        
        p=pop(i);
        
        popm(k).Position=Mutate(p.Position,VarRange);
        
        popm(k).Cost=CostFunction(popm(k).Position,y);
        
    end
    
    % Merge Population
    pop=[pop
         popc
         popm];
    
    % Sort Population
    [pop SortOrder]=SortPopulation(pop);
    
    % Delete Extra Individuals
    pop=pop(1:nPop);
    
    % Update Best Solution
    BestSol=pop(1);
    
    % Store Best Cost
    BestCost(it)=BestSol.Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end

%% Plots

figure;
plot(BestCost);

%convert from 2D to 3D
ClassifiedMap3D = Convert3d(pop(1).Position,h,w,nBand);
figure, imagesc(ClassifiedMap3D),title('Output GA Classified Map'),axis off
