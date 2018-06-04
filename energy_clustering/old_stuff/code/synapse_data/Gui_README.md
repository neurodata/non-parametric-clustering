
# Kristina15 data

## F0 
The file `k15F0_20710616.csv` holds the data matrix $X$ of integrated brighness features
across channels for each detected synapse location.

## Locations

The file `locations_k15F0_20170616.csv` holds the `(x,y,z)` locations
corresponding to the rows of $X$.

## MEDA run

I ran meda on `scaled(X, center = TRUE, scale = TRUE)` with results
here:
[http://docs.neurodata.io/synaptome-stats/Draft/Notes/Notes201706.html](http://docs.neurodata.io/synaptome-stats/Draft/Notes/Notes201706.html)
