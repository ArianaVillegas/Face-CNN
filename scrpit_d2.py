import Augmentor

p = Augmentor.Pipeline("./img/faces/") #poner ruta de las imagenes
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
p.sample(1000) #numero de imagenes finales que quieres tener
