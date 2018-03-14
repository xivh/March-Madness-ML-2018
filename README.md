# March-Madness-ML-2018
Google ML challenge for March Madness 2018

Some notes:
The players data is not clean, and has a lot of players double counted with slightly different names (600207 and 600208
in 2010, OOTESEY_ANTONIO and OOTSEY_ANTONIO). They could be removed with a similarity function (if two consecutive names
only have n letters different, remove one), but this would be a lot slower (you have to compare a bunch of names) and
would change the average name length for the team depending on if you got rid of the longer or shorter variant. It might
also remove some valid but similar names.

"TEAM" (60047 in 2010, etc.) shows up a lot, and I added an extra step to check and remove it.

Underscores aren't removed from names, but I think the regression can handle it.
