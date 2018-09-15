import string,cgi,time
import cgitb
cgitb.enable()
from os import curdir, sep
from http.server import BaseHTTPRequestHandler, HTTPServer
# from http.client import HTTPResponse
from Sickit_DNN import predict_blind
# from Sickit_RF import predict_blind
from xlsxwriter.utility import *
import pandas as pd
#import pri
import requests
import json

class MyHandler(BaseHTTPRequestHandler):

	def do_GET(self):
		if self.path=="/":
			self.path="/index.html"
		try:
			sendReply = False
			if self.path.endswith(".html"):
				mimetype='text/html'
				sendReply = True
			if self.path.endswith(".png"):
				mimetype='image/png'
				sendReply = True
			if self.path.endswith(".ico"):
				mimetype='image/x-icon'
				sendReply = True
			if self.path.endswith(".js"):
				mimetype='application/javascript'
				sendReply = True
			if self.path.endswith(".css"):
				mimetype='text/css'
				sendReply = True

			if sendReply == True:
				#Open the static file requested and send it
				f = open(curdir + sep + self.path, 'rb') 
				self.send_response(200)
				self.send_header('Content-type',mimetype)
				self.end_headers()
				self.wfile.write(f.read())
				f.close()
			return
				
		except IOError:
			self.send_error(404,'File Not Found: %s' % self.path)

	def do_POST(self):
		global rootnode
		try:
			ctype, pdict = cgi.parse_header(self.headers.get("Content-Type"))
			pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
			timeOut = 0.4
			if ctype == 'multipart/form-data':
				query = cgi.parse_multipart(self.rfile, pdict)

			upfilecontent = query.get('upfile')
			gReCaptcha = query.get('g-recaptcha-response')[0]
			url = 'https://www.google.com/recaptcha/api/siteverify'
			payload = {'secret': '6Lf4hCYUAAAAAKgdDH3j9Sr6-st2qI7awnuAnBbr', 'response': gReCaptcha}
			gReCaptchaResponse = requests.post(url, data=payload)
			gReCaptchaResponseJsonData = json.loads(gReCaptchaResponse.text)
			gReCaptchaResponseSuccess = gReCaptchaResponseJsonData["success"]
			if not gReCaptchaResponseSuccess:
				self.send_response(401)
			f = open('ToPredict.xlsx','wb')
			f.write(upfilecontent[0])
			f.close()
			# f = open('ToPredict.xlsx','r')
			result_rf , result_dnn = predict_blind(timeOut)
			[pdPreds_rf,pdIndices_rf] = code_to_string(result_rf)
			[pdPreds_dnn,pdIndices_dnn] = code_to_string(result_dnn)
			df = pd.DataFrame({'prediction_RF': pdPreds_rf,'prediction_DNN': pdPreds_dnn})
			writer = pd.ExcelWriter('predictions.xlsx', engine='xlsxwriter')
			df.to_excel(writer, sheet_name='predictions', index_label="Molecule Index")
			workbook = writer.book
			worksheet = writer.sheets['predictions']
			colsFormat = workbook.add_format({'align': 'center'})
			predsFormat = workbook.add_format({'align': 'center', 'border': 1})
			worksheet.set_column("A:C", 20, colsFormat)
			predsRange = xl_range(0, 2, len(result_rf), 1)
			worksheet.conditional_format(predsRange, {"type": "cell", "criteria": ">=", "value": "0", "format": predsFormat})
			writer.save()
			f = open('predictions.xlsx','rb')
			self.send_response(200)
			self.send_header('Content-type','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
			self.send_header('Content-Disposition', 'attachment; filename="BC_Prediction.xlsx"')
			self.end_headers()
			self.wfile.write(f.read())
		except:
			pass

def code_to_string(result):
	pdIndices = []
	pdPreds = []
	for i in range(1, (len(result) + 1)):
		pdIndices.append(i)
		if result[(i-1)] == 1:
			pdPreds.append("Bitter")
		else:
			pdPreds.append("Non-Bitter")
	return [pdPreds,pdIndices]


def main():
	try:
		server = HTTPServer(('', 80), MyHandler)
		print ('started httpserver...')
		print ('Listening on port 80')
		server.serve_forever()
	except KeyboardInterrupt:
		print ('^C received, shutting down server')
		server.socket.close()

if __name__ == '__main__':
	main()