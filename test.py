import numpy as np

my_list = ('Age-Young','Age-Adult','Age-Old', 'Gender-Female','Hair-Length-Short','Hair-Length-Long','Hair-Length-Bald','UpperBody-Length-Short',
 'UpperBody-Color-Black','UpperBody-Color-Blue','UpperBody-Color-Brown','UpperBody-Color-Green','UpperBody-Color-Grey',
 'UpperBody-Color-Orange','UpperBody-Color-Pink','UpperBody-Color-Purple','UpperBody-Color-Red','UpperBody-Color-White',
 'UpperBody-Color-Yellow','UpperBody-Color-Other','LowerBody-Length-Short','LowerBody-Color-Black','LowerBody-Color-Blue',
 'LowerBody-Color-Brown','LowerBody-Color-Green','LowerBody-Color-Grey','LowerBody-Color-Orange','LowerBody-Color-Pink',
 'LowerBody-Color-Purple','LowerBody-Color-Red','LowerBody-Color-White','LowerBody-Color-Yellow','LowerBody-Color-Other',
 'LowerBody-Type-Trousers&Shorts','LowerBody-Type-Skirt&Dress','Accessory-Backpack','Accessory-Bag',
 'Accessory-Glasses-Normal','Accessory-Glasses-Sun','Accessory-Hat')

# Elements to find
elements = ['LowerBody-Length-Short', 'LowerBody-Type-Trousers&Shorts', 'LowerBody-Type-Skirt&Dress']

print(len(my_list))
# Initialize a dictionary to hold the indices
indices_dict = {}

for element in elements:
    try:
        # Get the index of the first occurrence of the element
        index = my_list.index(element)
        indices_dict[element] = index
    except ValueError:
        # If the element is not found, store None or some indicator
        indices_dict[element] = None

print(indices_dict)