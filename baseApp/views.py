from django.shortcuts import render
import os
from joblib import load

# Create your views here.

def home(request):
   return render(request,'home.html')
def room(request):
    return render(request,'room.html')

# getting your joblib path
model_path = os.path.join(os.path.dirname(__file__), 'malaria_detector2.joblib')

# load the joblib file
trained_model = load(model_path)

def index(request):
	if request.method == "POST":
		data = []
		data.append(request.POST.get('temp'))
		data.append(request.POST.get('pdensity'))
		data.append(request.POST.get('wbcCount'))
        # data.append(request.POST.get('rbcCount'))
	    # data.append(request.POST.get('hblevel'))
		# data.append(request.POST.get('hematocrit'))
		data.append(request.POST.get('rbcCount'))
		data.append(request.POST.get('hblevel'))
		data.append(request.POST.get('hematocrit'))
		data.append(request.POST.get('mcVolume'))
		data.append(request.POST.get('mcHb'))
		data.append(request.POST.get('mcHbC'))
		data.append(request.POST.get('pcount'))
		data.append(request.POST.get('pdw'))
		data.append(request.POST.get('mpVl'))
		data.append(request.POST.get('npercent'))
		data.append(request.POST.get('lymphoPercent'))
		data.append(request.POST.get('mixedPercent'))
		data.append(request.POST.get('npCount'))
		data.append(request.POST.get('lymphoCount'))
		data.append(request.POST.get('mcCells'))

		prediction = trained_model.predict([data])
		return render(request, 'home.html', {'prediction': prediction })

		# if prediction == ['Pass ']:
		# 	return render(request, 'detection/index.html', {'prediction': 'You have barely passed this course'})
		# elif(prediction == ['Satisfaction ']):
		# 	return render(request, 'detection/index.html', {'prediction': prediction })
		# else:
		# 	return render(request, 'detection/index.html', {'prediction': 'Sorry to say but you failed the course!'})
#return render(request, 'home.html',{'prediction': prediction })