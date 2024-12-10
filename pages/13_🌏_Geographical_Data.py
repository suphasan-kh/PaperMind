import streamlit as st
from pymongo.mongo_client import MongoClient
from geopy.geocoders import Nominatim
import pandas as pd

gl = Nominatim(user_agent='ds')

uri = "mongodb://datasci:scopus888@datascidb.kanakornmek.dev:27017/?authSource=admin"

# Create a new client and connect to the server
client = MongoClient(uri)
db = client['datasci']
collection = db['papers']

# state = st.session_state
# default = ''
# st.sidebar.text_input('Search Author (Please Enter Full Name ex. Frodo Baggins)' ,value = default, key='dir')
# author = state.dir

state = st.session_state
if 'dir' not in state:
    state.dir = ''

st.sidebar.text_input('Search Author (Please Enter Full Name ex. Frodo Baggins)' ,value = state.dir, key='dir')
author = state.dir

# @st.cache_data
# def main():
if author.strip().count(' ') == 1:
    firstname, surname = author.strip().split()
    firstname = firstname.capitalize()
    surname = surname.capitalize()
    cursor = collection.find_one({'authors.author.preferred-name.ce:given-name':firstname ,'authors.author.preferred-name.ce:surname':surname})
    if cursor != None:
        city_id = set()
        coord = []
        cd = dict(cursor)
        authors = cd['authors']['author']
        for a in authors:
            if a['preferred-name']['ce:given-name'] == firstname and surname == a['preferred-name']['ce:surname']:
                affiliation_id = a['affiliation']
                break
        if type(affiliation_id).__name__ == 'dict':
            city_id.add(affiliation_id['@id'])
        else:
            for aff in affiliation_id:
                city_id.add(aff['@id'])

        affiliation = cd['affiliation']
        if type(affiliation).__name__ == 'dict':
            location = gl.geocode(affiliation['affiliation-city'] + ' ' + affiliation['affiliation-country'])
            coord.append([location.latitude,location.longitude,affiliation['affilname']])
        else:
            for af in affiliation:
                if af['@id'] in city_id:
                    location = gl.geocode(af['affiliation-city'] + ' ' + af['affiliation-country'])
                    coord.append([location.latitude,location.longitude,af['affilname']])

        coord = pd.DataFrame(coord)
        coord = coord.rename(columns = {0:'latitude',1:'longitude',2:'info'})
        
        st.write("# Affiliation Map")
        st.map(coord)
        st.write("# Affiliation Locations")
        st.write(coord)
        st.sidebar.write('Successful')
    else:
        st.sidebar.write('This person does not exist in database')
elif author.strip() == '':
    st.sidebar.write('')
else:
    st.sidebar.write('Wrong Format')

# main()