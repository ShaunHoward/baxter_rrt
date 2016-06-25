from django.db import models


class Classification(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=255)

    def __unicode__(self):
        return self.name


class Feature(models.Model):
    name = models.CharField(max_length=100)

    def __unicode__(self):
        return self.name


class ColorFeature(Feature):
    # use the colour package to parse this as a Color() object for better representation
    color = models.CharField(max_length=20)


class HandlingPlan(models.Model):
    # points will be json form of the Point() class with x, y, z and frame attributes
    points = models.TextField()

    def __unicode__(self):
        return self.points


class Image(models.Model):
    name = models.CharField(max_length=100)
    file = models.ImageField()

    def __unicode__(self):
        return self.name


class PointCloud(models.Model):
    # json serialization of Point() objects in a list
    points = models.TextField(default="{\"points\":[{\"x\":\"\", \"y\":\"\", \"z\":\"\"}]}")
    pcd = models.FileField()


class Object(models.Model):
    name = models.CharField(max_length=100)
    length = models.DecimalField(decimal_places=5, max_digits=8)
    width = models.DecimalField(decimal_places=5, max_digits=8)
    height = models.DecimalField(decimal_places=5, max_digits=8)
    point_clouds = models.ManyToManyField(PointCloud)
    images = models.ManyToManyField(Image)
    features = models.ManyToManyField(Feature)
    classifications = models.ManyToManyField(Classification)

    def __unicode__(self):
        return self.name

