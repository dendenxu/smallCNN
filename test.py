import re

url = "https://drscdn.500px.org/photo/161151147/q%3D80_m%3D2000_k%3D1/v2?sig=af8603cee4af0d9dee52bc41b8a0679c2237d19799e753a69d2a1a773ff2c962"
int(re.search("(?<=m%3D)\d*", url).group(0))
