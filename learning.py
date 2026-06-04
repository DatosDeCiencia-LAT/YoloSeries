

@app.get('/predict/realtime')

async def predict(wavelength: int = 193) -> JSONResponse:
    # Paso 1 — buscar imágenes disponibles
    result = Fido.search(
        a.Time(datetime.utcnow() - timedelta(hours=1), datetime.utcnow()),
        a.Instrument("AIA"),
        a.Wavelength(wavelength * u.angstrom),
    )
    
    if len(result) == 0:
        return JSONResponse(content={"message": "No se encontraron imágenes disponibles."}, status_code=404)

    # Paso 2 — descargar la más reciente
    downloaded = Fido.fetch(result[0, -1], path="/tmp/sdo/")
    image = sunpy.map.Map(downloaded[0]).data.astype(np.float32)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    detections = run_inference(image)
    log = log_prediction(source = "realtime", 
                         detections=detections)
    
    return JSONResponse(content=log)