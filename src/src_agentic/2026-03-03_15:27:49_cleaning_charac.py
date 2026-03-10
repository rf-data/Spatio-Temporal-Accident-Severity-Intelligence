import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    caracteristiques_2005_csv = pd.read_csv('caracteristiques_2005.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2005.csv': {e}")
caracteristiques_2005_csv['gps'] = caracteristiques_2005_csv['gps'].fillna(caracteristiques_2005_csv['gps'].median())
caracteristiques_2005_csv['lat'] = caracteristiques_2005_csv['lat'].fillna(caracteristiques_2005_csv['lat'].median())
caracteristiques_2005_csv['long'] = caracteristiques_2005_csv['long'].fillna(caracteristiques_2005_csv['long'].median())

# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2006_csv = pd.read_csv('caracteristiques_2006.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2006.csv': {e}")
caracteristiques_2006_csv['gps'] = caracteristiques_2006_csv['gps'].fillna(caracteristiques_2006_csv['gps'].median())
caracteristiques_2006_csv['lat'] = caracteristiques_2006_csv['lat'].fillna(caracteristiques_2006_csv['lat'].median())
caracteristiques_2006_csv['long'] = caracteristiques_2006_csv['long'].fillna(caracteristiques_2006_csv['long'].median())

# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2007_csv = pd.read_csv('caracteristiques_2007.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2007.csv': {e}")
caracteristiques_2007_csv['gps'] = caracteristiques_2007_csv['gps'].fillna(caracteristiques_2007_csv['gps'].median())
caracteristiques_2007_csv['lat'] = caracteristiques_2007_csv['lat'].fillna(caracteristiques_2007_csv['lat'].median())
caracteristiques_2007_csv['long'] = caracteristiques_2007_csv['long'].fillna(caracteristiques_2007_csv['long'].median())

# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2008_csv = pd.read_csv('caracteristiques_2008.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2008.csv': {e}")
caracteristiques_2008_csv['gps'] = caracteristiques_2008_csv['gps'].fillna(caracteristiques_2008_csv['gps'].median())
caracteristiques_2008_csv['lat'] = caracteristiques_2008_csv['lat'].fillna(caracteristiques_2008_csv['lat'].median())
caracteristiques_2008_csv['long'] = caracteristiques_2008_csv['long'].fillna(caracteristiques_2008_csv['long'].median())

# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2009_csv = pd.read_csv('caracteristiques_2009.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2009.csv': {e}")
caracteristiques_2009_csv['gps'] = caracteristiques_2009_csv['gps'].fillna(caracteristiques_2009_csv['gps'].median())
caracteristiques_2009_csv['lat'] = caracteristiques_2009_csv['lat'].fillna(caracteristiques_2009_csv['lat'].median())
caracteristiques_2009_csv['long'] = caracteristiques_2009_csv['long'].fillna(caracteristiques_2009_csv['long'].median())

# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2010_csv = pd.read_csv('caracteristiques_2010.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2010.csv': {e}")
caracteristiques_2010_csv['gps'] = caracteristiques_2010_csv['gps'].fillna(caracteristiques_2010_csv['gps'].median())
caracteristiques_2010_csv['lat'] = caracteristiques_2010_csv['lat'].fillna(caracteristiques_2010_csv['lat'].median())
caracteristiques_2010_csv['long'] = caracteristiques_2010_csv['long'].fillna(caracteristiques_2010_csv['long'].median())

# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2011_csv = pd.read_csv('caracteristiques_2011.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2011.csv': {e}")
caracteristiques_2011_csv['gps'] = caracteristiques_2011_csv['gps'].fillna(caracteristiques_2011_csv['gps'].median())
caracteristiques_2011_csv['lat'] = caracteristiques_2011_csv['lat'].fillna(caracteristiques_2011_csv['lat'].median())
caracteristiques_2011_csv['long'] = caracteristiques_2011_csv['long'].fillna(caracteristiques_2011_csv['long'].median())

# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2012_csv = pd.read_csv('caracteristiques_2012.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2012.csv': {e}")
caracteristiques_2012_csv['gps'] = caracteristiques_2012_csv['gps'].fillna(caracteristiques_2012_csv['gps'].median())
caracteristiques_2012_csv['lat'] = caracteristiques_2012_csv['lat'].fillna(caracteristiques_2012_csv['lat'].median())
caracteristiques_2012_csv['long'] = caracteristiques_2012_csv['long'].fillna(caracteristiques_2012_csv['long'].median())

# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2013_csv = pd.read_csv('caracteristiques_2013.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2013.csv': {e}")
caracteristiques_2013_csv['gps'] = caracteristiques_2013_csv['gps'].fillna(caracteristiques_2013_csv['gps'].median())
caracteristiques_2013_csv['lat'] = caracteristiques_2013_csv['lat'].fillna(caracteristiques_2013_csv['lat'].median())
caracteristiques_2013_csv['long'] = caracteristiques_2013_csv['long'].fillna(caracteristiques_2013_csv['long'].median())

# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2014_csv = pd.read_csv('caracteristiques_2014.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2014.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2015_csv = pd.read_csv('caracteristiques_2015.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2015.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques_2016_csv = pd.read_csv('caracteristiques_2016.csv')
except Exception as e:
    print("Error while loading 'caracteristiques_2016.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques-2017_csv = pd.read_csv('caracteristiques-2017.csv')
except Exception as e:
    print("Error while loading 'caracteristiques-2017.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques-2018_csv = pd.read_csv('caracteristiques-2018.csv')
except Exception as e:
    print("Error while loading 'caracteristiques-2018.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques-2019_csv = pd.read_csv('caracteristiques-2019.csv')
except Exception as e:
    print("Error while loading 'caracteristiques-2019.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques-2020_csv = pd.read_csv('caracteristiques-2020.csv')
except Exception as e:
    print("Error while loading 'caracteristiques-2020.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques-2021_csv = pd.read_csv('caracteristiques-2021.csv')
except Exception as e:
    print("Error while loading 'caracteristiques-2021.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caracteristiques-2022_csv = pd.read_csv('caracteristiques-2022.csv')
except Exception as e:
    print("Error while loading 'caracteristiques-2022.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caract-2023_csv = pd.read_csv('caract-2023.csv')
except Exception as e:
    print("Error while loading 'caract-2023.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'

try:
    caract-2024_csv = pd.read_csv('caract-2024.csv')
except Exception as e:
    print("Error while loading 'caract-2024.csv': {e}")
# WARNING: No handler implemented for 'handle_skewness'
# WARNING: No handler implemented for 'handle_kurtosis'
# WARNING: No handler implemented for 'cardinality_action'
