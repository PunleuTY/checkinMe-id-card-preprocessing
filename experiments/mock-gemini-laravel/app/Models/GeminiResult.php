<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class GeminiResult extends Model
{
    protected $guarded = ['created_at', 'updated_at'];

    public static function boot()
    {
        parent::boot();

        static::creating(function ($table) {
            if (auth()->check()) {
                $table->created_by_id = auth()->id();
                $table->created_by_type = get_class(auth()->user());
            }
        });
    }

    public function created_by()
    {
        return $this->morphTo();
    }
}