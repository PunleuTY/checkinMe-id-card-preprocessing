<?php

namespace App\Services;

use App\Models\GeminiResult;
use Illuminate\Support\Facades\Http;

class GeminiService extends RequestService
{
    public function __construct()
    {
        $this->base_url = config('gemini.base_url');
        $this->timeout = config('gemini.timeout');
    }

    public function scanNid($idImage, $request_from = null)
    {
        if (app()->environment() != 'production') {
            $data = [
                "idNumber" => "101325482",
                "lastNameKh" => "ជន",
                "firstNameKh" => "ពេងហុង",
                "dob" => "17/01/2001",
                "gender" => "M",
                "lastNameEn" => "CHORN",
                "firstNameEn" => "PENGHONG",
                "expiredDate" => "18/01/2030",
                "issuedDate" => "18/01/2020",
                "address" => "ភូមិកាច់ត្រក ឃុំលាយបូរ ស្រុកត្រាំកក់ តាកែវ",
                "pob" => "ឃុំលាយបូរ ស្រុកត្រាំកក់ តាកែវ",
                "MRZ1" => "IDKHM1013254824<<<<<<<<<<<<<<< ",
                "MRZ2" => "0101170M2610139KHM<<<<<<<<<<<8 ",
                "MRZ3" => "CHORN<<PENGHONG<<<<<<<<<<<<<<< "
            ];

            return [
                'success' => true,
                'data' => $data
            ];
        }

        if (strpos($idImage, 'base64,')) {
            $idImage = explode('base64,', $idImage)[1] ?? '';
        }

        $this->params = [
            'request_from' => $request_from
        ];

        $geminiResult = GeminiResult::create([
            'request_data' => json_encode($this->params),
        ]);

        $res = Http::withOptions(['verify' => false])
            ->timeout($this->timeout)
            ->attach('file', base64_decode($idImage), 'id_card.jpg')
            ->post($this->base_url . config('gemini.endpoints.scan_nid'), [
                'model' => config('gemini.model')
            ]);

        $json = $res->json();

        $geminiResult->update([
            'status_code' => $res->status(),
            'response_data' => json_encode($json),
        ]);

        if ($res->status() == 422) {
            return [
                'success' => false,
                'status' => 422,
                'message' => 'Image is invalid or cannot be processed.'
            ];
        } else if ($res->status() == 408) {
            return [
                'success' => false,
                'status' => 408,
                'message' => 'Request Timeout.'
            ];
        } else if ($res->status() != 200) {
            return [
                'success' => false,
                'status' => $res->status(),
                'message' => 'Something went wrong.'
            ];
        }

        return [
            'success' => true,
            'data' => $json['fields']
        ];
    }
}