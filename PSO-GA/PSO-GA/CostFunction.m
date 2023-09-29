function z=CostFunction(x,y)

    z=sqrt(sum((x-y).^2));

end