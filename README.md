# Python code to generate MOFs structure

INTRODUCTION: This script was part of the Ph.D. project where we
constructed MOFs consisting of Metallic (Fe, Ru, Cu) ions coordinated 
with bipyridine molecules. This technical construction consists of 
first creating the main unit fully optimized with DFT in the gas phase 
and getting the corresponding NBO or Mulliken charges. The next phase 
is to repeat this unit in all three directions with the same corresponding 
charges of the main unit. Overall, this is the assumption where we have
assumed that the system is homogenous, the electrostatic field is the same 
everywhere. Now, we have to fix the boundary points. This, however, is a 
difficult problem because the truncated charges have to be taken into account. 
This is explained in the thesis. 

![alt text](https://user-images.githubusercontent.com/7361722/71469992-1d33cd80-27cb-11ea-9a45-2b2552446ce3.png)

This complex construction was part of a PhD thesis in which we devise a method to construct Fe(bpy) based MOFs. It uses python libraries. The Amber FF parameters were fitted to DFT calculations. [Click this link to get to HAL archive!](https://hal.archives-ouvertes.fr/tel-02058650)
