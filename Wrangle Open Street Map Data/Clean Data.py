
# coding: utf-8

# In[1]:

#import necessary packages

import xml.etree.ElementTree as ET
import pprint
import re
import codecs
import json
import pymongo
from pymongo import MongoClient
import json
from bson import json_util as ju
import pandas as pd


# In[2]:

#Files

OSM_FILE = "chennai_india.osm"  
SAMPLE_FILE = "chennai_sample.osm"


# In[3]:

#db connection
client=MongoClient("mongodb://localhost:27017")
db=client.sample


# In[4]:

#used to load Created and Location
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]
pos = ["lat", "lon"]


# In[5]:

# list to fix Street Abbreviation issues
mapping_key = ["st", "st.", "ave", "rd.", "rd", "extn", "extn.","St", "St.", "Ave", "Rd.", "Rd", "Extn", "Extn.",
              "ST", "ST.", "AVE", "RD.", "RD", "EXTN", "EXTN."]
mapping = { "st": "Street",
            "st.": "Street",
            "ave" : "Avenue",
            "rd." : "Road",
            "rd" : "Road",
            "extn" : "Extension",
            "extn." : "Extension",
           "St": "Street",
            "St.": "Street",
            "Ave" : "Avenue",
            "Rd." : "Road",
            "Rd" : "Road",
            "Extn" : "Extension",
            "Extn." : "Extension",
           "ST": "Street",
            "ST.": "Street",
            "AVE" : "Avenue",
            "RD." : "Road",
            "RD" : "Road",
            "EXTN" : "Extension",
            "EXTN." : "Extension"
            }


# In[6]:

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')


# In[7]:

def update_street(v_attrib):
    sp = v_attrib.split()
    if sp[len(sp)-1] in mapping_key:
        street_update = re.sub(sp[len(sp)-1], mapping[sp[len(sp)-1]], v_attrib).title()
    else:
        street_update = v_attrib.title() 
    return street_update


# In[8]:

def update_city(v_attrib):
    if not ',' in v_attrib and 'chennai' in v_attrib.lower():
        city_update = 'Chennai'
        # Below step is to fix the City issue with chennai, format
    elif 'chennai,' in v_attrib.lower():
        city_update = 'Chennai'
    else:
        city_update = v_attrib.title()
    
    return city_update


# In[9]:

def update_postcode(v_attrib):
    postcode_update = v_attrib.replace(" ", "") 
    return postcode_update


# In[10]:

def shape_element(element):
    node = {}
    created = {}
    pos_list = [0.00,0.00]
    refs = []
    addr = {}
    data_check = {}
    if element.tag == "node" or element.tag == "way" :
        for i in element.attrib:
            if i in CREATED:
                created[i] = element.attrib[i]
                
            elif i in pos:
                if i == 'lat':
                    pos_list[0] =float(element.attrib[i])
                else:
                    pos_list[1] =float(element.attrib[i])
            else:
                node[i] = element.attrib[i]
        node['created'] = created
        if pos_list != [0.00,0.00]:
            node['pos'] = pos_list
        
        for child in element:
            if child.tag == 'tag':
                k_val = child.attrib['k']
                if problemchars.search(k_val):
                    pass
                else:
                    if child.attrib['k'].startswith('addr:'):
                        if child.attrib['k'].count(':') < 2:
                            # Below step is to fix the City issue (checking for string without comma, as many lines have valid names with comma)
                            if child.attrib['k'][5:] == 'city':
                                addr[child.attrib['k'][5:]] = update_city(child.attrib['v']) 
                                #data_check['Old City'] = child.attrib['v']
                                #data_check['New City'] = update_city(child.attrib['v']) 
                            elif child.attrib['k'][5:] == 'postcode':
                                addr[child.attrib['k'][5:]] = update_postcode(child.attrib['v'])
                                #data_check['Old Postcode'] = child.attrib['v']
                                #data_check['New Postcode'] = update_postcode(child.attrib['v']) 
                            # Below step is to fix the Street abbreviations like St, RD etc.,
                            elif child.attrib['k'][5:] == 'street':
                                addr[child.attrib['k'][5:]] = update_street(child.attrib['v'])
                                #data_check['Old Street'] = child.attrib['v']
                                #data_check['New Street'] = update_street(child.attrib['v']) 
                            else:
                                addr[child.attrib['k'][5:]] = child.attrib['v'].title()
                            a = 'yes'
                    else:
                        node[child.attrib['k']] = child.attrib['v']
            elif child.tag == 'nd':
                refs.append(child.attrib['ref'])
            else:
                pass
        if addr:
            node['address'] = addr
        if refs:
            node['node_refs'] = refs
        node['type'] = element.tag    
        return node
    else:
        return None


# In[11]:

def process_map(file_in, pretty = False):
    # You do not need to change this file
    file_out = "{0}.json".format(file_in)
    data = []
    data_test = []
    dataframe = pd.DataFrame([])
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            el1 = shape_element_test(element)
            if el:
                data.append(el)
                dataframe.append(el1)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data


# In[39]:

def final():
       
    data = process_map('chennai_india.osm', False)

if __name__ == "__main__":
    final()

