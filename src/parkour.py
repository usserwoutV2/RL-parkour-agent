import sys
import os

current_dir = os.path.dirname(__file__)
sys.path.insert( 0, current_dir)
from geneticAlgorithm.Genetics import Genetics

#{
#  "clients": Client[]
#  "maxActionCount": int, max amount of steps to finish the parkour
#  "goal": {x:int, y:int, z:int}
#}
def parkour(options):
  
  genetics = Genetics(options)
  genetics.reset()

  while True:
    yield 
    #print(str(genetics))
    genetics.step()
    

  
    
    
    
    

    
  
      

  