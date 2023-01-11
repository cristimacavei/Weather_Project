import en_core_web_lg
import pandas as pd
import os
from tqdm import tqdm
from spacy.tokens import DocBin
import spacy

nlp = spacy.load("en_core_web_lg")


text = "A car is parked in the snow"
doc = nlp(text)

train = [
    ("Climate change could make giant hailstones more common, research suggests.",{"entities":[(32,43,"hail")]}),
    ("In most regions, hailstorm severity is expected to increase with climate change",{"entities":[(17,27,"hail")]}),
    ("Dew is the moisture that forms at night when objects or the ground outside cool down by radiating, or emitting, their heat",{"entities":[(0,4,"dew"),(11,20,"dew")]}),
    ("Dew is a natural form of water, formed as water vapor condenses. Dew, like the glistening drops on this grassy field in Anaconda, Montana, forms as water near the surface of the ground is cooled to its dew point, the temperature at which water vapor condenses. The dew point varies by area and even time of day.",{"entities":[(0,4,"dew"),(25,30,"dew"),(42,53,"dew"),(65,68,"dew"),(90,96,"dew"),(148,154,"dew"),(178,185,"dew"),(202,211,"dew"),(238,249,"dew"),(250,259,"dew"),(265,274,"dew")]}),
    ("Fog is a cloud that touches the ground.",{"entities":[(0,4,"fogsmog"),(9,15,"fogsmog")]}),
    ("Fog shrouds the dense equatorial rain forest in Rio Muni, Guinea. Many rain forest trees absorb water from fog as well as precipitation.",{"entities":[(0,3,"fogsmog"),(4,12,"fogsmog"),(96,102,"fogsmog"),(107,111,"fogsmog")]}),
    ("Frost is water vapor, or water in gas form, that becomes solid. Frost usually forms on objects like cars, windows, and plants that are outside in air that is saturated, or filled, with moisture.",{"entities":[(0,5,"frost"),(9,20,"frost"),(25,31,"frost"),(34,42,"frost"),(57,62,"frost"),(64,70,"frost"),(100,104,"frost"),(106,113,"frost"),(119,126,"frost")]}),
    ("Frost dusts leaves of a plant in Bruges, Belgium. This type of frost, called radiation frost or hoarfrost, develops as objects (such as this plant) become colder than the surrounding air.",{"entities":[(0,6,"frost"),(63,68,"frost"),(77,92,"frost"),(96,105,"frost"),(155,162,"frost")]}),
    ("Glaze is ice formed by freezing precipitation covering the ground or exposed objects.",{"entities":[(0,6,"glaze"),(9,12,"glaze"),(23,45,"glaze"),(59,66,"glaze"),(77,84,"glaze")]}),
    ("Glaze or glaze ice, also called glazed frost, is a smooth, transparent and homogeneous ice coating occurring when freezing rain or drizzle hits a surface.It is similar in appearance to clear ice, which forms from supercooled water droplets.",{"entities":[(0,6,"glaze"),(9,18,"glaze"),(32,44,"glaze"),(87,98,"glaze"),(114,127,"glaze"),(131,139,"glaze"),(185,194,"glaze"),(225,239,"glaze")]}),
    # ("Lightning is an electric charge or current. It can come from the clouds to the ground, from cloud to cloud, or from the ground to a cloud.",{"entities":[(0,10,"lightning"),(16,24,"lightning"),(25,32,"lightning"),(35,42,"lightning"),(16,32,"lightning"),(65,72,"lightning")]}),
("Lightning is an electric charge or current. It can come from the clouds to the ground, from cloud to cloud, or from the ground to a cloud.",{"entities":[(0,9,"lightning"),(16,24,"lightning"),(25,31,"lightning"),(35,42,"lightning")]}),
    ("Lightning strikes the ground near Hong Kong. Lightning is an electrical charge, or current. It can travel from the clouds to the ground, from cloud to cloud, or from the ground to a cloud.",{"entities":[(0,9,"lightning"),(10,17,"lightning"),(45,54,"lightning"),(61,71,"lightning"),(72,78,"lightning"),(83,90,"lightning")]}),
    ("Rain is liquid precipitation: water falling from the sky. Raindrops fall to Earth when clouds become saturated, or filled, with water droplets.",{"entities":[(0,5,"rain"),(8,28,"rain"),(30,36,"rain"),(53,56,"rain"),(58,68,"rain"),(128,142,"rain")]}),
    ("Rain falls along a Wyoming freeway. Rain falls at different rates in different parts of the world. Dry desert regions can get less than a centimeter of rain every year, while tropical rain forests receive more than a meter. Wyoming is a dry state, with much of the land receiving fewer than 250 millimeters (10 inches) of rain every year.",{"entities":[(0,5,"rain"),(36,46,"rain"),(5,11,"rain"),(152,157,"rain"),(184,196,"rain"),(322,327,"rain")]}),
    ("A rainbow is a multicolored arc made by light striking water droplets.",{"entities":[(2,10,"rainbow"),(15,28,"rainbow"),(28,32,"rainbow"),(55,69,"rainbow")]}),
    ("A rainbow is produced by a ray of light being refracted and reflected by water. Light is refracted (bent) as it enters a water droplet. It is then reflected (bounced off) the back of the droplet.",{"entities":[(2,10,"rainbow"),(27,39,"rainbow"),(46,56,"rainbow"),(60,70,"rainbow"),(73,78,"rainbow"),(80,86,"rainbow"),(89,99,"rainbow"),(121,134,"rainbow"),(147,157,"rainbow"),(187,194,"rainbow")]}),
    ("By definition, rime is a deposit of interlocking ice crystals formed by direct sublimation on objects, usually those of small diameter freely exposed to the air, such as tree branches.",{"entities":[(15,20,"rime"),(25,33,"rime"),(36,61,"rime")]}),
    ("rime, white, opaque, granular deposit of ice crystals formed on objects that are at a temperature below the freezing point. Rime occurs when supercooled water droplets (at a temperature lower than 0 C [32 F]) in fog come in contact with a surface that is also at a temperature below freezing;",{"entities":[(0,4,"rime"),(21,37,"rime"),(41,53,"rime"),(108,122,"rime"),(124,129,"rime"),(153,167,"rime")]}),
    ("Sand and dust storms usually occur when strong winds lift large amounts of sand and dust from bare, dry soils into the atmosphere.",{"entities":[(0,5,"sandstorm"),(14,21,"sandstorm"),(75,80,"sandstorm"),(9,14,"sandstorm"),(84,89,"sandstorm"),(104,110,"sandstorm")]}),
    ("Sand and dust storms are common meteorological hazards in arid and semi-arid regions. They are usually caused by thunderstorms or strong pressure gradients associated with cyclones which increase wind speed over a wide area.",{"entities":[(0,5,"sandstorm"),(9,14,"sandstorm"),(14,21,"sandstorm"),(113,127,"sandstorm"),(172,181,"sandstorm"),(196,201,"sandstorm")]}),
    ("snow, the solid form of water that crystallizes in the atmosphere and, falling to the Earth, covers, permanently or temporarily, about 23 percent of the Earths surface.",{"entities":[(0,4,"snow"),(24,30,"snow"),(10,20,"snow"),(35,48,"snow")]}),
    ("Snowflakes are formed by crystals of ice that generally have a hexagonal pattern, often beautifully intricate. The size and shape of the crystals depend mainly on the temperature and the amount of water vapour available as they develop.",{"entities":[(0,11,"snow"),(197,209,"snow"),(137,146,"snow"),(25,40,"snow")]}),
]

# label_list = en_core_web_lg["train"].features[f"ner_tags"].feature.names
# print(label_list)
# train = [
#
# ("An average-sized strawberry has about 200 seeds on its outer surface and are quite edible.",{"entities":[(17,27,"Fruit")]}),
#           ("The outer skin of Guava is bitter tasting and thick, dark green for raw fruits and as the fruit ripens, the bitterness subsides. ",{"entities":[(18,23,"Fruit")]}),
#           ("Grapes are one of the most widely grown types of fruits in the world, chiefly for the making of different wines. ",{"entities":[(0,6,"Fruit")]}),
#           ("Watermelon is composed of 92 percent water and significant amounts of Vitamins and antioxidants. ",{"entities":[(0,10,"Fruit")]}),
#           ("Papaya fruits are usually cylindrical in shape and the size can go beyond 20 inches. ",{"entities":[(0,6,"Fruit")]}),
#           ("Mango, the King of the fruits is a drupe fruit that grows in tropical regions. ",{"entities":[(0,5,"Fruit")]}),
#           ("undefined",{"entities":[(0,6,"Fruit")]}),
#           ("Oranges are great source of vitamin C",{"entities":[(0,7,"Fruit")]}),
#           ("A apple a day keeps doctor away. ",{"entities":[(2,7,"Fruit")]})
# ]

db = DocBin() # create a DocBin object

for text, annot in tqdm(train): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    print("Ents: ", ents)
    doc.ents = ents # label the text with the ents
    db.add(doc)

db.to_disk("./train.spacy") # save the docbin object