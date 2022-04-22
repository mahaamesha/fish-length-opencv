import vs
from shapely.geometry import Polygon
from ladybug_geometry_polyskel.polyskel import skeleton_as_edge_list

def Create_Center_Polys_Skel(p):
    #create a blank array to hold the of the shape we wish to
    #    find the centerline for
    list = []
    #Loop through each vertex in the shape and add its coordinates
    #    to the list
    for currentPoint in range(vs.GetVertNum(p)):
        x1, y1 = vs.GetPolyPt(p, currentPoint+1)
        list.append([x1, y1])
    #Create a "shapely" object of polygon from the vaules in the 
    #    list we have just generated
    polygon = Polygon(list)
    #Run the "Labybug PolySkel" function on that polygon and assign
    #    the return vaule, to "skelcenterline"
    skelcenterline = skeleton_as_edge_list(list)
    #Loop through each created vertex in "skelcenterline" and extract
    #    the start coordinates and end coordinates of each line that was
    #    created. When the start and end coordinates have been separated
    #    move the Vectorworks "cursor" to the start point of the line,
    #    and draw a line terminating at the end point
    for m in skelcenterline:
        startcord, endcord = m
        startx, starty = startcord
        endx, endy = endcord
        vs.MoveTo(startx, starty)
        vs.LineTo(endx, endy)
    #Run the menu command "compose" on all created objects to try and 
    #    consolidate them into as few polylines as possible for manipulation
    #    later
    vs.DoMenuTextByName('Compose',0);


#Set "selectedPoly" to be the currently selected Vectorworks polygon
selectedPoly = vs.FSActLayer()
#Run "Create_Center_Polys_Skel" passing it the polygon we just created
#    from the selected poly in Vectorworks
Create_Center_Polys_Skel(selectedPoly)