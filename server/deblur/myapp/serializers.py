from rest_framework import serializers

class GetImageUploadSerializers(serializers.Serializer):
    image = serializers.ImageField()
