from django.db import models


class Classification(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=255)

    def __unicode__(self):
        return self.name


class Feature(models.Model):
    name = models.CharField(max_length=100)
    # points will be json form of the Point() class with x, y, z and frame attributes
    points = models.TextField()

    def __unicode__(self):
        return self.name


class ColorFeature(Feature):
    color = models.CharField()


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


class Object(models.Model):
    name = models.CharField(max_length=100)
    length = models.DecimalField()
    width = models.DecimalField()
    height = models.DecimalField()
    point_clouds = models.ManyToManyField(PointCloud)
    images = models.ManyToManyField(Image)
    features = models.ManyToManyField(Feature)
    classifications = models.ManyToManyField(Classification)

    def __unicode__(self):
        return self.name


class PointCloud(models.Model):
    file = models.FileField()
