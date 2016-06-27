from __future__ import absolute_import

from django.contrib import admin

from .models import *


admin.site.register(Classification)
admin.site.register(ColorFeature)
admin.site.register(HandlingPlan)
admin.site.register(Image)
admin.site.register(PointCloud)
admin.site.register(Object)

