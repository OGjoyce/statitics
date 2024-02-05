reset;


#definir conjuntos

set U = {"San Cristobal", "Carretera al Salvador", "Zona 1", "Zona 16"}; #Especificar de donde es el origen
set E = {"Villa Nueva","San Lucas","Zona 17","Zona 2"}; #entregas
set K = {"Bikini Electrico", "Bikini Menta", "Bikini TieDye" }; #producto

#definir parametros
param C{U,E}; #Costo de transporte desde un origen U hasta el punto de destino e por cada unidad de producto k
param D{K,E}; #Demanda del producto k en el punto de destino e.
param J{K,U}; #Capacidad de almacenamiento del producto k en la ubicación U 



#definimos variables
var xuek{U,E,K} >= 0; #Número total de productos K a entregar desde el origen U hasta el punto de entrega e.


#funciones objetivo
minimize func_obj: sum{k in K} sum{u in U} sum{e in E} (C[u,e]*xuek[u,e,k]);

#restricciones
subject to demanda{k in K, e in E}:  sum{u in U}xuek[u,e,k]  >=  D[k,e]; 
subject to almacenamiento{k in K,v in U}:  sum{e in E}xuek[v,e,k] <= J[k,v];
#restriccion para amarrar los costos y la demanda?
#subject to oferta_demanda{kk in K, u in U}: sum{k in K}sum{e in E}D[k,e] <= J[kk,u];


data 'C:\Users\cmont\proyecto.dat';
solve;

display xuek, D;
#display almacenamiento;
#display D;






