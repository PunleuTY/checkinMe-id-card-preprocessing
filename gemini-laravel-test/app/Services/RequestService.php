<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;

class RequestService
{
    protected $method;
    protected $params;
    protected $key_param; // json, multipart
    protected $base_url;
    protected $end_point;
    protected $headers = [];
    protected $timeout = 10;

    public function get()
    {
        return Http::withOptions(['verify' => false])
            ->timeout($this->timeout)
            ->send($this->method, $this->base_url . $this->end_point, [
                'headers' => $this->headers,
                $this->key_param => $this->params
            ]);
    }

    public function getAsJson()
    {
        return Http::asJson()->timeout($this->timeout)->send($this->method, $this->base_url . $this->end_point, [
            'headers' => $this->headers,
            'body' => json_encode($this->params)
        ]);
    }

    public function request($is_array = false)
    {
        $res = $this->get();
        if ($is_array) {
            return $res->json();
        }

        return $res->object();
    }

    public function toArray()
    {
        return $this->request(true);
    }
}
