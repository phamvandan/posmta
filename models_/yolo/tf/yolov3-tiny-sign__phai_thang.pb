
{
net1Placeholder*&
shape:˙˙˙˙˙˙˙˙˙  *
dtype0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙  
š
6convolutional1/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@convolutional1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ŗ
4convolutional1/kernel/Initializer/random_uniform/minConst*(
_class
loc:@convolutional1/kernel*
valueB
 *Đ?ž*
dtype0*
_output_shapes
: 
Ŗ
4convolutional1/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@convolutional1/kernel*
valueB
 *Đ?>*
dtype0*
_output_shapes
: 

>convolutional1/kernel/Initializer/random_uniform/RandomUniformRandomUniform6convolutional1/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@convolutional1/kernel*
dtype0*
seed2 *&
_output_shapes
:
ō
4convolutional1/kernel/Initializer/random_uniform/subSub4convolutional1/kernel/Initializer/random_uniform/max4convolutional1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional1/kernel*
_output_shapes
: 

4convolutional1/kernel/Initializer/random_uniform/mulMul>convolutional1/kernel/Initializer/random_uniform/RandomUniform4convolutional1/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@convolutional1/kernel*&
_output_shapes
:
ū
0convolutional1/kernel/Initializer/random_uniformAdd4convolutional1/kernel/Initializer/random_uniform/mul4convolutional1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional1/kernel*&
_output_shapes
:
Ã
convolutional1/kernel
VariableV2*
shape:*
shared_name *(
_class
loc:@convolutional1/kernel*
dtype0*
	container *&
_output_shapes
:
ķ
convolutional1/kernel/AssignAssignconvolutional1/kernel0convolutional1/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@convolutional1/kernel*
validate_shape(*&
_output_shapes
:

convolutional1/kernel/readIdentityconvolutional1/kernel*
T0*(
_class
loc:@convolutional1/kernel*&
_output_shapes
:
m
convolutional1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional1/Conv2DConv2Dnet1convolutional1/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙  
¯
/convolutional1/BatchNorm/gamma/Initializer/onesConst*1
_class'
%#loc:@convolutional1/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes
:
Ŋ
convolutional1/BatchNorm/gamma
VariableV2*
shape:*
shared_name *1
_class'
%#loc:@convolutional1/BatchNorm/gamma*
dtype0*
	container *
_output_shapes
:

%convolutional1/BatchNorm/gamma/AssignAssignconvolutional1/BatchNorm/gamma/convolutional1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*1
_class'
%#loc:@convolutional1/BatchNorm/gamma*
validate_shape(*
_output_shapes
:
§
#convolutional1/BatchNorm/gamma/readIdentityconvolutional1/BatchNorm/gamma*
T0*1
_class'
%#loc:@convolutional1/BatchNorm/gamma*
_output_shapes
:
Ž
/convolutional1/BatchNorm/beta/Initializer/zerosConst*0
_class&
$"loc:@convolutional1/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes
:
ģ
convolutional1/BatchNorm/beta
VariableV2*
shape:*
shared_name *0
_class&
$"loc:@convolutional1/BatchNorm/beta*
dtype0*
	container *
_output_shapes
:
ū
$convolutional1/BatchNorm/beta/AssignAssignconvolutional1/BatchNorm/beta/convolutional1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@convolutional1/BatchNorm/beta*
validate_shape(*
_output_shapes
:
¤
"convolutional1/BatchNorm/beta/readIdentityconvolutional1/BatchNorm/beta*
T0*0
_class&
$"loc:@convolutional1/BatchNorm/beta*
_output_shapes
:
ŧ
6convolutional1/BatchNorm/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@convolutional1/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
É
$convolutional1/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *7
_class-
+)loc:@convolutional1/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes
:

+convolutional1/BatchNorm/moving_mean/AssignAssign$convolutional1/BatchNorm/moving_mean6convolutional1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@convolutional1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:
š
)convolutional1/BatchNorm/moving_mean/readIdentity$convolutional1/BatchNorm/moving_mean*
T0*7
_class-
+)loc:@convolutional1/BatchNorm/moving_mean*
_output_shapes
:
Ã
9convolutional1/BatchNorm/moving_variance/Initializer/onesConst*;
_class1
/-loc:@convolutional1/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes
:
Ņ
(convolutional1/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *;
_class1
/-loc:@convolutional1/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes
:
Š
/convolutional1/BatchNorm/moving_variance/AssignAssign(convolutional1/BatchNorm/moving_variance9convolutional1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@convolutional1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:
Å
-convolutional1/BatchNorm/moving_variance/readIdentity(convolutional1/BatchNorm/moving_variance*
T0*;
_class1
/-loc:@convolutional1/BatchNorm/moving_variance*
_output_shapes
:

)convolutional1/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional1/Conv2D#convolutional1/BatchNorm/gamma/read"convolutional1/BatchNorm/beta/read)convolutional1/BatchNorm/moving_mean/read-convolutional1/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *M
_output_shapes;
9:˙˙˙˙˙˙˙˙˙  :::::
c
convolutional1/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional1/Activation	LeakyRelu)convolutional1/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*1
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ä
maxpool1/MaxPoolMaxPoolconvolutional1/Activation*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĐĐ
š
6convolutional2/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@convolutional2/kernel*%
valueB"             *
dtype0*
_output_shapes
:
Ŗ
4convolutional2/kernel/Initializer/random_uniform/minConst*(
_class
loc:@convolutional2/kernel*
valueB
 *ī[ņŊ*
dtype0*
_output_shapes
: 
Ŗ
4convolutional2/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@convolutional2/kernel*
valueB
 *ī[ņ=*
dtype0*
_output_shapes
: 

>convolutional2/kernel/Initializer/random_uniform/RandomUniformRandomUniform6convolutional2/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@convolutional2/kernel*
dtype0*
seed2 *&
_output_shapes
: 
ō
4convolutional2/kernel/Initializer/random_uniform/subSub4convolutional2/kernel/Initializer/random_uniform/max4convolutional2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional2/kernel*
_output_shapes
: 

4convolutional2/kernel/Initializer/random_uniform/mulMul>convolutional2/kernel/Initializer/random_uniform/RandomUniform4convolutional2/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@convolutional2/kernel*&
_output_shapes
: 
ū
0convolutional2/kernel/Initializer/random_uniformAdd4convolutional2/kernel/Initializer/random_uniform/mul4convolutional2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional2/kernel*&
_output_shapes
: 
Ã
convolutional2/kernel
VariableV2*
shape: *
shared_name *(
_class
loc:@convolutional2/kernel*
dtype0*
	container *&
_output_shapes
: 
ķ
convolutional2/kernel/AssignAssignconvolutional2/kernel0convolutional2/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@convolutional2/kernel*
validate_shape(*&
_output_shapes
: 

convolutional2/kernel/readIdentityconvolutional2/kernel*
T0*(
_class
loc:@convolutional2/kernel*&
_output_shapes
: 
m
convolutional2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional2/Conv2DConv2Dmaxpool1/MaxPoolconvolutional2/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĐĐ 
¯
/convolutional2/BatchNorm/gamma/Initializer/onesConst*1
_class'
%#loc:@convolutional2/BatchNorm/gamma*
valueB *  ?*
dtype0*
_output_shapes
: 
Ŋ
convolutional2/BatchNorm/gamma
VariableV2*
shape: *
shared_name *1
_class'
%#loc:@convolutional2/BatchNorm/gamma*
dtype0*
	container *
_output_shapes
: 

%convolutional2/BatchNorm/gamma/AssignAssignconvolutional2/BatchNorm/gamma/convolutional2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*1
_class'
%#loc:@convolutional2/BatchNorm/gamma*
validate_shape(*
_output_shapes
: 
§
#convolutional2/BatchNorm/gamma/readIdentityconvolutional2/BatchNorm/gamma*
T0*1
_class'
%#loc:@convolutional2/BatchNorm/gamma*
_output_shapes
: 
Ž
/convolutional2/BatchNorm/beta/Initializer/zerosConst*0
_class&
$"loc:@convolutional2/BatchNorm/beta*
valueB *    *
dtype0*
_output_shapes
: 
ģ
convolutional2/BatchNorm/beta
VariableV2*
shape: *
shared_name *0
_class&
$"loc:@convolutional2/BatchNorm/beta*
dtype0*
	container *
_output_shapes
: 
ū
$convolutional2/BatchNorm/beta/AssignAssignconvolutional2/BatchNorm/beta/convolutional2/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@convolutional2/BatchNorm/beta*
validate_shape(*
_output_shapes
: 
¤
"convolutional2/BatchNorm/beta/readIdentityconvolutional2/BatchNorm/beta*
T0*0
_class&
$"loc:@convolutional2/BatchNorm/beta*
_output_shapes
: 
ŧ
6convolutional2/BatchNorm/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@convolutional2/BatchNorm/moving_mean*
valueB *    *
dtype0*
_output_shapes
: 
É
$convolutional2/BatchNorm/moving_mean
VariableV2*
shape: *
shared_name *7
_class-
+)loc:@convolutional2/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes
: 

+convolutional2/BatchNorm/moving_mean/AssignAssign$convolutional2/BatchNorm/moving_mean6convolutional2/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@convolutional2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
: 
š
)convolutional2/BatchNorm/moving_mean/readIdentity$convolutional2/BatchNorm/moving_mean*
T0*7
_class-
+)loc:@convolutional2/BatchNorm/moving_mean*
_output_shapes
: 
Ã
9convolutional2/BatchNorm/moving_variance/Initializer/onesConst*;
_class1
/-loc:@convolutional2/BatchNorm/moving_variance*
valueB *  ?*
dtype0*
_output_shapes
: 
Ņ
(convolutional2/BatchNorm/moving_variance
VariableV2*
shape: *
shared_name *;
_class1
/-loc:@convolutional2/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes
: 
Š
/convolutional2/BatchNorm/moving_variance/AssignAssign(convolutional2/BatchNorm/moving_variance9convolutional2/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@convolutional2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
: 
Å
-convolutional2/BatchNorm/moving_variance/readIdentity(convolutional2/BatchNorm/moving_variance*
T0*;
_class1
/-loc:@convolutional2/BatchNorm/moving_variance*
_output_shapes
: 

)convolutional2/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional2/Conv2D#convolutional2/BatchNorm/gamma/read"convolutional2/BatchNorm/beta/read)convolutional2/BatchNorm/moving_mean/read-convolutional2/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *M
_output_shapes;
9:˙˙˙˙˙˙˙˙˙ĐĐ : : : : :
c
convolutional2/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional2/Activation	LeakyRelu)convolutional2/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ĐĐ 
Â
maxpool2/MaxPoolMaxPoolconvolutional2/Activation*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙hh 
š
6convolutional3/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@convolutional3/kernel*%
valueB"          @   *
dtype0*
_output_shapes
:
Ŗ
4convolutional3/kernel/Initializer/random_uniform/minConst*(
_class
loc:@convolutional3/kernel*
valueB
 *ĢĒĒŊ*
dtype0*
_output_shapes
: 
Ŗ
4convolutional3/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@convolutional3/kernel*
valueB
 *ĢĒĒ=*
dtype0*
_output_shapes
: 

>convolutional3/kernel/Initializer/random_uniform/RandomUniformRandomUniform6convolutional3/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@convolutional3/kernel*
dtype0*
seed2 *&
_output_shapes
: @
ō
4convolutional3/kernel/Initializer/random_uniform/subSub4convolutional3/kernel/Initializer/random_uniform/max4convolutional3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional3/kernel*
_output_shapes
: 

4convolutional3/kernel/Initializer/random_uniform/mulMul>convolutional3/kernel/Initializer/random_uniform/RandomUniform4convolutional3/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@convolutional3/kernel*&
_output_shapes
: @
ū
0convolutional3/kernel/Initializer/random_uniformAdd4convolutional3/kernel/Initializer/random_uniform/mul4convolutional3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional3/kernel*&
_output_shapes
: @
Ã
convolutional3/kernel
VariableV2*
shape: @*
shared_name *(
_class
loc:@convolutional3/kernel*
dtype0*
	container *&
_output_shapes
: @
ķ
convolutional3/kernel/AssignAssignconvolutional3/kernel0convolutional3/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@convolutional3/kernel*
validate_shape(*&
_output_shapes
: @

convolutional3/kernel/readIdentityconvolutional3/kernel*
T0*(
_class
loc:@convolutional3/kernel*&
_output_shapes
: @
m
convolutional3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional3/Conv2DConv2Dmaxpool2/MaxPoolconvolutional3/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙hh@
¯
/convolutional3/BatchNorm/gamma/Initializer/onesConst*1
_class'
%#loc:@convolutional3/BatchNorm/gamma*
valueB@*  ?*
dtype0*
_output_shapes
:@
Ŋ
convolutional3/BatchNorm/gamma
VariableV2*
shape:@*
shared_name *1
_class'
%#loc:@convolutional3/BatchNorm/gamma*
dtype0*
	container *
_output_shapes
:@

%convolutional3/BatchNorm/gamma/AssignAssignconvolutional3/BatchNorm/gamma/convolutional3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*1
_class'
%#loc:@convolutional3/BatchNorm/gamma*
validate_shape(*
_output_shapes
:@
§
#convolutional3/BatchNorm/gamma/readIdentityconvolutional3/BatchNorm/gamma*
T0*1
_class'
%#loc:@convolutional3/BatchNorm/gamma*
_output_shapes
:@
Ž
/convolutional3/BatchNorm/beta/Initializer/zerosConst*0
_class&
$"loc:@convolutional3/BatchNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
ģ
convolutional3/BatchNorm/beta
VariableV2*
shape:@*
shared_name *0
_class&
$"loc:@convolutional3/BatchNorm/beta*
dtype0*
	container *
_output_shapes
:@
ū
$convolutional3/BatchNorm/beta/AssignAssignconvolutional3/BatchNorm/beta/convolutional3/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@convolutional3/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
¤
"convolutional3/BatchNorm/beta/readIdentityconvolutional3/BatchNorm/beta*
T0*0
_class&
$"loc:@convolutional3/BatchNorm/beta*
_output_shapes
:@
ŧ
6convolutional3/BatchNorm/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@convolutional3/BatchNorm/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
É
$convolutional3/BatchNorm/moving_mean
VariableV2*
shape:@*
shared_name *7
_class-
+)loc:@convolutional3/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes
:@

+convolutional3/BatchNorm/moving_mean/AssignAssign$convolutional3/BatchNorm/moving_mean6convolutional3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@convolutional3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:@
š
)convolutional3/BatchNorm/moving_mean/readIdentity$convolutional3/BatchNorm/moving_mean*
T0*7
_class-
+)loc:@convolutional3/BatchNorm/moving_mean*
_output_shapes
:@
Ã
9convolutional3/BatchNorm/moving_variance/Initializer/onesConst*;
_class1
/-loc:@convolutional3/BatchNorm/moving_variance*
valueB@*  ?*
dtype0*
_output_shapes
:@
Ņ
(convolutional3/BatchNorm/moving_variance
VariableV2*
shape:@*
shared_name *;
_class1
/-loc:@convolutional3/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes
:@
Š
/convolutional3/BatchNorm/moving_variance/AssignAssign(convolutional3/BatchNorm/moving_variance9convolutional3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@convolutional3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
Å
-convolutional3/BatchNorm/moving_variance/readIdentity(convolutional3/BatchNorm/moving_variance*
T0*;
_class1
/-loc:@convolutional3/BatchNorm/moving_variance*
_output_shapes
:@

)convolutional3/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional3/Conv2D#convolutional3/BatchNorm/gamma/read"convolutional3/BatchNorm/beta/read)convolutional3/BatchNorm/moving_mean/read-convolutional3/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *K
_output_shapes9
7:˙˙˙˙˙˙˙˙˙hh@:@:@:@:@:
c
convolutional3/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional3/Activation	LeakyRelu)convolutional3/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*/
_output_shapes
:˙˙˙˙˙˙˙˙˙hh@
Â
maxpool3/MaxPoolMaxPoolconvolutional3/Activation*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙44@
š
6convolutional4/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@convolutional4/kernel*%
valueB"      @      *
dtype0*
_output_shapes
:
Ŗ
4convolutional4/kernel/Initializer/random_uniform/minConst*(
_class
loc:@convolutional4/kernel*
valueB
 *ī[qŊ*
dtype0*
_output_shapes
: 
Ŗ
4convolutional4/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@convolutional4/kernel*
valueB
 *ī[q=*
dtype0*
_output_shapes
: 

>convolutional4/kernel/Initializer/random_uniform/RandomUniformRandomUniform6convolutional4/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@convolutional4/kernel*
dtype0*
seed2 *'
_output_shapes
:@
ō
4convolutional4/kernel/Initializer/random_uniform/subSub4convolutional4/kernel/Initializer/random_uniform/max4convolutional4/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional4/kernel*
_output_shapes
: 

4convolutional4/kernel/Initializer/random_uniform/mulMul>convolutional4/kernel/Initializer/random_uniform/RandomUniform4convolutional4/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@convolutional4/kernel*'
_output_shapes
:@
˙
0convolutional4/kernel/Initializer/random_uniformAdd4convolutional4/kernel/Initializer/random_uniform/mul4convolutional4/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional4/kernel*'
_output_shapes
:@
Å
convolutional4/kernel
VariableV2*
shape:@*
shared_name *(
_class
loc:@convolutional4/kernel*
dtype0*
	container *'
_output_shapes
:@
ô
convolutional4/kernel/AssignAssignconvolutional4/kernel0convolutional4/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@convolutional4/kernel*
validate_shape(*'
_output_shapes
:@

convolutional4/kernel/readIdentityconvolutional4/kernel*
T0*(
_class
loc:@convolutional4/kernel*'
_output_shapes
:@
m
convolutional4/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional4/Conv2DConv2Dmaxpool3/MaxPoolconvolutional4/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙44
ą
/convolutional4/BatchNorm/gamma/Initializer/onesConst*1
_class'
%#loc:@convolutional4/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ŋ
convolutional4/BatchNorm/gamma
VariableV2*
shape:*
shared_name *1
_class'
%#loc:@convolutional4/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:

%convolutional4/BatchNorm/gamma/AssignAssignconvolutional4/BatchNorm/gamma/convolutional4/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*1
_class'
%#loc:@convolutional4/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:
¨
#convolutional4/BatchNorm/gamma/readIdentityconvolutional4/BatchNorm/gamma*
T0*1
_class'
%#loc:@convolutional4/BatchNorm/gamma*
_output_shapes	
:
°
/convolutional4/BatchNorm/beta/Initializer/zerosConst*0
_class&
$"loc:@convolutional4/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes	
:
Ŋ
convolutional4/BatchNorm/beta
VariableV2*
shape:*
shared_name *0
_class&
$"loc:@convolutional4/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:
˙
$convolutional4/BatchNorm/beta/AssignAssignconvolutional4/BatchNorm/beta/convolutional4/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@convolutional4/BatchNorm/beta*
validate_shape(*
_output_shapes	
:
Ĩ
"convolutional4/BatchNorm/beta/readIdentityconvolutional4/BatchNorm/beta*
T0*0
_class&
$"loc:@convolutional4/BatchNorm/beta*
_output_shapes	
:
ž
6convolutional4/BatchNorm/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@convolutional4/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
Ë
$convolutional4/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *7
_class-
+)loc:@convolutional4/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:

+convolutional4/BatchNorm/moving_mean/AssignAssign$convolutional4/BatchNorm/moving_mean6convolutional4/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@convolutional4/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:
ē
)convolutional4/BatchNorm/moving_mean/readIdentity$convolutional4/BatchNorm/moving_mean*
T0*7
_class-
+)loc:@convolutional4/BatchNorm/moving_mean*
_output_shapes	
:
Å
9convolutional4/BatchNorm/moving_variance/Initializer/onesConst*;
_class1
/-loc:@convolutional4/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
Ķ
(convolutional4/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *;
_class1
/-loc:@convolutional4/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:
Ē
/convolutional4/BatchNorm/moving_variance/AssignAssign(convolutional4/BatchNorm/moving_variance9convolutional4/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@convolutional4/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:
Æ
-convolutional4/BatchNorm/moving_variance/readIdentity(convolutional4/BatchNorm/moving_variance*
T0*;
_class1
/-loc:@convolutional4/BatchNorm/moving_variance*
_output_shapes	
:

)convolutional4/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional4/Conv2D#convolutional4/BatchNorm/gamma/read"convolutional4/BatchNorm/beta/read)convolutional4/BatchNorm/moving_mean/read-convolutional4/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙44:::::
c
convolutional4/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional4/Activation	LeakyRelu)convolutional4/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*0
_output_shapes
:˙˙˙˙˙˙˙˙˙44
Ã
maxpool4/MaxPoolMaxPoolconvolutional4/Activation*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
6convolutional5/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@convolutional5/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ŗ
4convolutional5/kernel/Initializer/random_uniform/minConst*(
_class
loc:@convolutional5/kernel*
valueB
 *ĢĒ*Ŋ*
dtype0*
_output_shapes
: 
Ŗ
4convolutional5/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@convolutional5/kernel*
valueB
 *ĢĒ*=*
dtype0*
_output_shapes
: 

>convolutional5/kernel/Initializer/random_uniform/RandomUniformRandomUniform6convolutional5/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@convolutional5/kernel*
dtype0*
seed2 *(
_output_shapes
:
ō
4convolutional5/kernel/Initializer/random_uniform/subSub4convolutional5/kernel/Initializer/random_uniform/max4convolutional5/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional5/kernel*
_output_shapes
: 

4convolutional5/kernel/Initializer/random_uniform/mulMul>convolutional5/kernel/Initializer/random_uniform/RandomUniform4convolutional5/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@convolutional5/kernel*(
_output_shapes
:

0convolutional5/kernel/Initializer/random_uniformAdd4convolutional5/kernel/Initializer/random_uniform/mul4convolutional5/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional5/kernel*(
_output_shapes
:
Į
convolutional5/kernel
VariableV2*
shape:*
shared_name *(
_class
loc:@convolutional5/kernel*
dtype0*
	container *(
_output_shapes
:
õ
convolutional5/kernel/AssignAssignconvolutional5/kernel0convolutional5/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@convolutional5/kernel*
validate_shape(*(
_output_shapes
:

convolutional5/kernel/readIdentityconvolutional5/kernel*
T0*(
_class
loc:@convolutional5/kernel*(
_output_shapes
:
m
convolutional5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional5/Conv2DConv2Dmaxpool4/MaxPoolconvolutional5/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
/convolutional5/BatchNorm/gamma/Initializer/onesConst*1
_class'
%#loc:@convolutional5/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ŋ
convolutional5/BatchNorm/gamma
VariableV2*
shape:*
shared_name *1
_class'
%#loc:@convolutional5/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:

%convolutional5/BatchNorm/gamma/AssignAssignconvolutional5/BatchNorm/gamma/convolutional5/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*1
_class'
%#loc:@convolutional5/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:
¨
#convolutional5/BatchNorm/gamma/readIdentityconvolutional5/BatchNorm/gamma*
T0*1
_class'
%#loc:@convolutional5/BatchNorm/gamma*
_output_shapes	
:
°
/convolutional5/BatchNorm/beta/Initializer/zerosConst*0
_class&
$"loc:@convolutional5/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes	
:
Ŋ
convolutional5/BatchNorm/beta
VariableV2*
shape:*
shared_name *0
_class&
$"loc:@convolutional5/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:
˙
$convolutional5/BatchNorm/beta/AssignAssignconvolutional5/BatchNorm/beta/convolutional5/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@convolutional5/BatchNorm/beta*
validate_shape(*
_output_shapes	
:
Ĩ
"convolutional5/BatchNorm/beta/readIdentityconvolutional5/BatchNorm/beta*
T0*0
_class&
$"loc:@convolutional5/BatchNorm/beta*
_output_shapes	
:
ž
6convolutional5/BatchNorm/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@convolutional5/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
Ë
$convolutional5/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *7
_class-
+)loc:@convolutional5/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:

+convolutional5/BatchNorm/moving_mean/AssignAssign$convolutional5/BatchNorm/moving_mean6convolutional5/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@convolutional5/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:
ē
)convolutional5/BatchNorm/moving_mean/readIdentity$convolutional5/BatchNorm/moving_mean*
T0*7
_class-
+)loc:@convolutional5/BatchNorm/moving_mean*
_output_shapes	
:
Å
9convolutional5/BatchNorm/moving_variance/Initializer/onesConst*;
_class1
/-loc:@convolutional5/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
Ķ
(convolutional5/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *;
_class1
/-loc:@convolutional5/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:
Ē
/convolutional5/BatchNorm/moving_variance/AssignAssign(convolutional5/BatchNorm/moving_variance9convolutional5/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@convolutional5/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:
Æ
-convolutional5/BatchNorm/moving_variance/readIdentity(convolutional5/BatchNorm/moving_variance*
T0*;
_class1
/-loc:@convolutional5/BatchNorm/moving_variance*
_output_shapes	
:

)convolutional5/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional5/Conv2D#convolutional5/BatchNorm/gamma/read"convolutional5/BatchNorm/beta/read)convolutional5/BatchNorm/moving_mean/read-convolutional5/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::
c
convolutional5/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional5/Activation	LeakyRelu)convolutional5/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ã
maxpool5/MaxPoolMaxPoolconvolutional5/Activation*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
6convolutional6/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@convolutional6/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ŗ
4convolutional6/kernel/Initializer/random_uniform/minConst*(
_class
loc:@convolutional6/kernel*
valueB
 *ī[ņŧ*
dtype0*
_output_shapes
: 
Ŗ
4convolutional6/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@convolutional6/kernel*
valueB
 *ī[ņ<*
dtype0*
_output_shapes
: 

>convolutional6/kernel/Initializer/random_uniform/RandomUniformRandomUniform6convolutional6/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@convolutional6/kernel*
dtype0*
seed2 *(
_output_shapes
:
ō
4convolutional6/kernel/Initializer/random_uniform/subSub4convolutional6/kernel/Initializer/random_uniform/max4convolutional6/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional6/kernel*
_output_shapes
: 

4convolutional6/kernel/Initializer/random_uniform/mulMul>convolutional6/kernel/Initializer/random_uniform/RandomUniform4convolutional6/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@convolutional6/kernel*(
_output_shapes
:

0convolutional6/kernel/Initializer/random_uniformAdd4convolutional6/kernel/Initializer/random_uniform/mul4convolutional6/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional6/kernel*(
_output_shapes
:
Į
convolutional6/kernel
VariableV2*
shape:*
shared_name *(
_class
loc:@convolutional6/kernel*
dtype0*
	container *(
_output_shapes
:
õ
convolutional6/kernel/AssignAssignconvolutional6/kernel0convolutional6/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@convolutional6/kernel*
validate_shape(*(
_output_shapes
:

convolutional6/kernel/readIdentityconvolutional6/kernel*
T0*(
_class
loc:@convolutional6/kernel*(
_output_shapes
:
m
convolutional6/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional6/Conv2DConv2Dmaxpool5/MaxPoolconvolutional6/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
/convolutional6/BatchNorm/gamma/Initializer/onesConst*1
_class'
%#loc:@convolutional6/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ŋ
convolutional6/BatchNorm/gamma
VariableV2*
shape:*
shared_name *1
_class'
%#loc:@convolutional6/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:

%convolutional6/BatchNorm/gamma/AssignAssignconvolutional6/BatchNorm/gamma/convolutional6/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*1
_class'
%#loc:@convolutional6/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:
¨
#convolutional6/BatchNorm/gamma/readIdentityconvolutional6/BatchNorm/gamma*
T0*1
_class'
%#loc:@convolutional6/BatchNorm/gamma*
_output_shapes	
:
°
/convolutional6/BatchNorm/beta/Initializer/zerosConst*0
_class&
$"loc:@convolutional6/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes	
:
Ŋ
convolutional6/BatchNorm/beta
VariableV2*
shape:*
shared_name *0
_class&
$"loc:@convolutional6/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:
˙
$convolutional6/BatchNorm/beta/AssignAssignconvolutional6/BatchNorm/beta/convolutional6/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@convolutional6/BatchNorm/beta*
validate_shape(*
_output_shapes	
:
Ĩ
"convolutional6/BatchNorm/beta/readIdentityconvolutional6/BatchNorm/beta*
T0*0
_class&
$"loc:@convolutional6/BatchNorm/beta*
_output_shapes	
:
ž
6convolutional6/BatchNorm/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@convolutional6/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
Ë
$convolutional6/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *7
_class-
+)loc:@convolutional6/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:

+convolutional6/BatchNorm/moving_mean/AssignAssign$convolutional6/BatchNorm/moving_mean6convolutional6/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@convolutional6/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:
ē
)convolutional6/BatchNorm/moving_mean/readIdentity$convolutional6/BatchNorm/moving_mean*
T0*7
_class-
+)loc:@convolutional6/BatchNorm/moving_mean*
_output_shapes	
:
Å
9convolutional6/BatchNorm/moving_variance/Initializer/onesConst*;
_class1
/-loc:@convolutional6/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
Ķ
(convolutional6/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *;
_class1
/-loc:@convolutional6/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:
Ē
/convolutional6/BatchNorm/moving_variance/AssignAssign(convolutional6/BatchNorm/moving_variance9convolutional6/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@convolutional6/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:
Æ
-convolutional6/BatchNorm/moving_variance/readIdentity(convolutional6/BatchNorm/moving_variance*
T0*;
_class1
/-loc:@convolutional6/BatchNorm/moving_variance*
_output_shapes	
:

)convolutional6/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional6/Conv2D#convolutional6/BatchNorm/gamma/read"convolutional6/BatchNorm/beta/read)convolutional6/BatchNorm/moving_mean/read-convolutional6/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::
c
convolutional6/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional6/Activation	LeakyRelu)convolutional6/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ã
maxpool6/MaxPoolMaxPoolconvolutional6/Activation*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
6convolutional7/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@convolutional7/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ŗ
4convolutional7/kernel/Initializer/random_uniform/minConst*(
_class
loc:@convolutional7/kernel*
valueB
 *ĢĒĒŧ*
dtype0*
_output_shapes
: 
Ŗ
4convolutional7/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@convolutional7/kernel*
valueB
 *ĢĒĒ<*
dtype0*
_output_shapes
: 

>convolutional7/kernel/Initializer/random_uniform/RandomUniformRandomUniform6convolutional7/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@convolutional7/kernel*
dtype0*
seed2 *(
_output_shapes
:
ō
4convolutional7/kernel/Initializer/random_uniform/subSub4convolutional7/kernel/Initializer/random_uniform/max4convolutional7/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional7/kernel*
_output_shapes
: 

4convolutional7/kernel/Initializer/random_uniform/mulMul>convolutional7/kernel/Initializer/random_uniform/RandomUniform4convolutional7/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@convolutional7/kernel*(
_output_shapes
:

0convolutional7/kernel/Initializer/random_uniformAdd4convolutional7/kernel/Initializer/random_uniform/mul4convolutional7/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional7/kernel*(
_output_shapes
:
Į
convolutional7/kernel
VariableV2*
shape:*
shared_name *(
_class
loc:@convolutional7/kernel*
dtype0*
	container *(
_output_shapes
:
õ
convolutional7/kernel/AssignAssignconvolutional7/kernel0convolutional7/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@convolutional7/kernel*
validate_shape(*(
_output_shapes
:

convolutional7/kernel/readIdentityconvolutional7/kernel*
T0*(
_class
loc:@convolutional7/kernel*(
_output_shapes
:
m
convolutional7/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional7/Conv2DConv2Dmaxpool6/MaxPoolconvolutional7/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŋ
?convolutional7/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*1
_class'
%#loc:@convolutional7/BatchNorm/gamma*
valueB:*
dtype0*
_output_shapes
:
­
5convolutional7/BatchNorm/gamma/Initializer/ones/ConstConst*1
_class'
%#loc:@convolutional7/BatchNorm/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 

/convolutional7/BatchNorm/gamma/Initializer/onesFill?convolutional7/BatchNorm/gamma/Initializer/ones/shape_as_tensor5convolutional7/BatchNorm/gamma/Initializer/ones/Const*
T0*1
_class'
%#loc:@convolutional7/BatchNorm/gamma*

index_type0*
_output_shapes	
:
ŋ
convolutional7/BatchNorm/gamma
VariableV2*
shape:*
shared_name *1
_class'
%#loc:@convolutional7/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:

%convolutional7/BatchNorm/gamma/AssignAssignconvolutional7/BatchNorm/gamma/convolutional7/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*1
_class'
%#loc:@convolutional7/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:
¨
#convolutional7/BatchNorm/gamma/readIdentityconvolutional7/BatchNorm/gamma*
T0*1
_class'
%#loc:@convolutional7/BatchNorm/gamma*
_output_shapes	
:
ŧ
?convolutional7/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@convolutional7/BatchNorm/beta*
valueB:*
dtype0*
_output_shapes
:
Ŧ
5convolutional7/BatchNorm/beta/Initializer/zeros/ConstConst*0
_class&
$"loc:@convolutional7/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 

/convolutional7/BatchNorm/beta/Initializer/zerosFill?convolutional7/BatchNorm/beta/Initializer/zeros/shape_as_tensor5convolutional7/BatchNorm/beta/Initializer/zeros/Const*
T0*0
_class&
$"loc:@convolutional7/BatchNorm/beta*

index_type0*
_output_shapes	
:
Ŋ
convolutional7/BatchNorm/beta
VariableV2*
shape:*
shared_name *0
_class&
$"loc:@convolutional7/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:
˙
$convolutional7/BatchNorm/beta/AssignAssignconvolutional7/BatchNorm/beta/convolutional7/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@convolutional7/BatchNorm/beta*
validate_shape(*
_output_shapes	
:
Ĩ
"convolutional7/BatchNorm/beta/readIdentityconvolutional7/BatchNorm/beta*
T0*0
_class&
$"loc:@convolutional7/BatchNorm/beta*
_output_shapes	
:
Ę
Fconvolutional7/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*7
_class-
+)loc:@convolutional7/BatchNorm/moving_mean*
valueB:*
dtype0*
_output_shapes
:
ē
<convolutional7/BatchNorm/moving_mean/Initializer/zeros/ConstConst*7
_class-
+)loc:@convolutional7/BatchNorm/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
ĩ
6convolutional7/BatchNorm/moving_mean/Initializer/zerosFillFconvolutional7/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor<convolutional7/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*7
_class-
+)loc:@convolutional7/BatchNorm/moving_mean*

index_type0*
_output_shapes	
:
Ë
$convolutional7/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *7
_class-
+)loc:@convolutional7/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:

+convolutional7/BatchNorm/moving_mean/AssignAssign$convolutional7/BatchNorm/moving_mean6convolutional7/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@convolutional7/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:
ē
)convolutional7/BatchNorm/moving_mean/readIdentity$convolutional7/BatchNorm/moving_mean*
T0*7
_class-
+)loc:@convolutional7/BatchNorm/moving_mean*
_output_shapes	
:
Ņ
Iconvolutional7/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*;
_class1
/-loc:@convolutional7/BatchNorm/moving_variance*
valueB:*
dtype0*
_output_shapes
:
Á
?convolutional7/BatchNorm/moving_variance/Initializer/ones/ConstConst*;
_class1
/-loc:@convolutional7/BatchNorm/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Â
9convolutional7/BatchNorm/moving_variance/Initializer/onesFillIconvolutional7/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor?convolutional7/BatchNorm/moving_variance/Initializer/ones/Const*
T0*;
_class1
/-loc:@convolutional7/BatchNorm/moving_variance*

index_type0*
_output_shapes	
:
Ķ
(convolutional7/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *;
_class1
/-loc:@convolutional7/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:
Ē
/convolutional7/BatchNorm/moving_variance/AssignAssign(convolutional7/BatchNorm/moving_variance9convolutional7/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@convolutional7/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:
Æ
-convolutional7/BatchNorm/moving_variance/readIdentity(convolutional7/BatchNorm/moving_variance*
T0*;
_class1
/-loc:@convolutional7/BatchNorm/moving_variance*
_output_shapes	
:

)convolutional7/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional7/Conv2D#convolutional7/BatchNorm/gamma/read"convolutional7/BatchNorm/beta/read)convolutional7/BatchNorm/moving_mean/read-convolutional7/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::
c
convolutional7/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional7/Activation	LeakyRelu)convolutional7/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
6convolutional8/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@convolutional8/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ŗ
4convolutional8/kernel/Initializer/random_uniform/minConst*(
_class
loc:@convolutional8/kernel*
valueB
 *7Ŋ*
dtype0*
_output_shapes
: 
Ŗ
4convolutional8/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@convolutional8/kernel*
valueB
 *7=*
dtype0*
_output_shapes
: 

>convolutional8/kernel/Initializer/random_uniform/RandomUniformRandomUniform6convolutional8/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@convolutional8/kernel*
dtype0*
seed2 *(
_output_shapes
:
ō
4convolutional8/kernel/Initializer/random_uniform/subSub4convolutional8/kernel/Initializer/random_uniform/max4convolutional8/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional8/kernel*
_output_shapes
: 

4convolutional8/kernel/Initializer/random_uniform/mulMul>convolutional8/kernel/Initializer/random_uniform/RandomUniform4convolutional8/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@convolutional8/kernel*(
_output_shapes
:

0convolutional8/kernel/Initializer/random_uniformAdd4convolutional8/kernel/Initializer/random_uniform/mul4convolutional8/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional8/kernel*(
_output_shapes
:
Į
convolutional8/kernel
VariableV2*
shape:*
shared_name *(
_class
loc:@convolutional8/kernel*
dtype0*
	container *(
_output_shapes
:
õ
convolutional8/kernel/AssignAssignconvolutional8/kernel0convolutional8/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@convolutional8/kernel*
validate_shape(*(
_output_shapes
:

convolutional8/kernel/readIdentityconvolutional8/kernel*
T0*(
_class
loc:@convolutional8/kernel*(
_output_shapes
:
m
convolutional8/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional8/Conv2DConv2Dconvolutional7/Activationconvolutional8/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
/convolutional8/BatchNorm/gamma/Initializer/onesConst*1
_class'
%#loc:@convolutional8/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ŋ
convolutional8/BatchNorm/gamma
VariableV2*
shape:*
shared_name *1
_class'
%#loc:@convolutional8/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:

%convolutional8/BatchNorm/gamma/AssignAssignconvolutional8/BatchNorm/gamma/convolutional8/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*1
_class'
%#loc:@convolutional8/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:
¨
#convolutional8/BatchNorm/gamma/readIdentityconvolutional8/BatchNorm/gamma*
T0*1
_class'
%#loc:@convolutional8/BatchNorm/gamma*
_output_shapes	
:
°
/convolutional8/BatchNorm/beta/Initializer/zerosConst*0
_class&
$"loc:@convolutional8/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes	
:
Ŋ
convolutional8/BatchNorm/beta
VariableV2*
shape:*
shared_name *0
_class&
$"loc:@convolutional8/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:
˙
$convolutional8/BatchNorm/beta/AssignAssignconvolutional8/BatchNorm/beta/convolutional8/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@convolutional8/BatchNorm/beta*
validate_shape(*
_output_shapes	
:
Ĩ
"convolutional8/BatchNorm/beta/readIdentityconvolutional8/BatchNorm/beta*
T0*0
_class&
$"loc:@convolutional8/BatchNorm/beta*
_output_shapes	
:
ž
6convolutional8/BatchNorm/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@convolutional8/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
Ë
$convolutional8/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *7
_class-
+)loc:@convolutional8/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:

+convolutional8/BatchNorm/moving_mean/AssignAssign$convolutional8/BatchNorm/moving_mean6convolutional8/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@convolutional8/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:
ē
)convolutional8/BatchNorm/moving_mean/readIdentity$convolutional8/BatchNorm/moving_mean*
T0*7
_class-
+)loc:@convolutional8/BatchNorm/moving_mean*
_output_shapes	
:
Å
9convolutional8/BatchNorm/moving_variance/Initializer/onesConst*;
_class1
/-loc:@convolutional8/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
Ķ
(convolutional8/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *;
_class1
/-loc:@convolutional8/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:
Ē
/convolutional8/BatchNorm/moving_variance/AssignAssign(convolutional8/BatchNorm/moving_variance9convolutional8/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@convolutional8/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:
Æ
-convolutional8/BatchNorm/moving_variance/readIdentity(convolutional8/BatchNorm/moving_variance*
T0*;
_class1
/-loc:@convolutional8/BatchNorm/moving_variance*
_output_shapes	
:

)convolutional8/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional8/Conv2D#convolutional8/BatchNorm/gamma/read"convolutional8/BatchNorm/beta/read)convolutional8/BatchNorm/moving_mean/read-convolutional8/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::
c
convolutional8/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional8/Activation	LeakyRelu)convolutional8/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
6convolutional9/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@convolutional9/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ŗ
4convolutional9/kernel/Initializer/random_uniform/minConst*(
_class
loc:@convolutional9/kernel*
valueB
 *ī[ņŧ*
dtype0*
_output_shapes
: 
Ŗ
4convolutional9/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@convolutional9/kernel*
valueB
 *ī[ņ<*
dtype0*
_output_shapes
: 

>convolutional9/kernel/Initializer/random_uniform/RandomUniformRandomUniform6convolutional9/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@convolutional9/kernel*
dtype0*
seed2 *(
_output_shapes
:
ō
4convolutional9/kernel/Initializer/random_uniform/subSub4convolutional9/kernel/Initializer/random_uniform/max4convolutional9/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional9/kernel*
_output_shapes
: 

4convolutional9/kernel/Initializer/random_uniform/mulMul>convolutional9/kernel/Initializer/random_uniform/RandomUniform4convolutional9/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@convolutional9/kernel*(
_output_shapes
:

0convolutional9/kernel/Initializer/random_uniformAdd4convolutional9/kernel/Initializer/random_uniform/mul4convolutional9/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@convolutional9/kernel*(
_output_shapes
:
Į
convolutional9/kernel
VariableV2*
shape:*
shared_name *(
_class
loc:@convolutional9/kernel*
dtype0*
	container *(
_output_shapes
:
õ
convolutional9/kernel/AssignAssignconvolutional9/kernel0convolutional9/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@convolutional9/kernel*
validate_shape(*(
_output_shapes
:

convolutional9/kernel/readIdentityconvolutional9/kernel*
T0*(
_class
loc:@convolutional9/kernel*(
_output_shapes
:
m
convolutional9/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional9/Conv2DConv2Dconvolutional8/Activationconvolutional9/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
/convolutional9/BatchNorm/gamma/Initializer/onesConst*1
_class'
%#loc:@convolutional9/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ŋ
convolutional9/BatchNorm/gamma
VariableV2*
shape:*
shared_name *1
_class'
%#loc:@convolutional9/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:

%convolutional9/BatchNorm/gamma/AssignAssignconvolutional9/BatchNorm/gamma/convolutional9/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*1
_class'
%#loc:@convolutional9/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:
¨
#convolutional9/BatchNorm/gamma/readIdentityconvolutional9/BatchNorm/gamma*
T0*1
_class'
%#loc:@convolutional9/BatchNorm/gamma*
_output_shapes	
:
°
/convolutional9/BatchNorm/beta/Initializer/zerosConst*0
_class&
$"loc:@convolutional9/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes	
:
Ŋ
convolutional9/BatchNorm/beta
VariableV2*
shape:*
shared_name *0
_class&
$"loc:@convolutional9/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:
˙
$convolutional9/BatchNorm/beta/AssignAssignconvolutional9/BatchNorm/beta/convolutional9/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@convolutional9/BatchNorm/beta*
validate_shape(*
_output_shapes	
:
Ĩ
"convolutional9/BatchNorm/beta/readIdentityconvolutional9/BatchNorm/beta*
T0*0
_class&
$"loc:@convolutional9/BatchNorm/beta*
_output_shapes	
:
ž
6convolutional9/BatchNorm/moving_mean/Initializer/zerosConst*7
_class-
+)loc:@convolutional9/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
Ë
$convolutional9/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *7
_class-
+)loc:@convolutional9/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:

+convolutional9/BatchNorm/moving_mean/AssignAssign$convolutional9/BatchNorm/moving_mean6convolutional9/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@convolutional9/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:
ē
)convolutional9/BatchNorm/moving_mean/readIdentity$convolutional9/BatchNorm/moving_mean*
T0*7
_class-
+)loc:@convolutional9/BatchNorm/moving_mean*
_output_shapes	
:
Å
9convolutional9/BatchNorm/moving_variance/Initializer/onesConst*;
_class1
/-loc:@convolutional9/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
Ķ
(convolutional9/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *;
_class1
/-loc:@convolutional9/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:
Ē
/convolutional9/BatchNorm/moving_variance/AssignAssign(convolutional9/BatchNorm/moving_variance9convolutional9/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*;
_class1
/-loc:@convolutional9/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:
Æ
-convolutional9/BatchNorm/moving_variance/readIdentity(convolutional9/BatchNorm/moving_variance*
T0*;
_class1
/-loc:@convolutional9/BatchNorm/moving_variance*
_output_shapes	
:

)convolutional9/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional9/Conv2D#convolutional9/BatchNorm/gamma/read"convolutional9/BatchNorm/beta/read)convolutional9/BatchNorm/moving_mean/read-convolutional9/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::
c
convolutional9/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional9/Activation	LeakyRelu)convolutional9/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ģ
7convolutional10/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@convolutional10/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ĩ
5convolutional10/kernel/Initializer/random_uniform/minConst*)
_class
loc:@convolutional10/kernel*
valueB
 *JŲŊ*
dtype0*
_output_shapes
: 
Ĩ
5convolutional10/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@convolutional10/kernel*
valueB
 *JŲ=*
dtype0*
_output_shapes
: 

?convolutional10/kernel/Initializer/random_uniform/RandomUniformRandomUniform7convolutional10/kernel/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@convolutional10/kernel*
dtype0*
seed2 *'
_output_shapes
:
ö
5convolutional10/kernel/Initializer/random_uniform/subSub5convolutional10/kernel/Initializer/random_uniform/max5convolutional10/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@convolutional10/kernel*
_output_shapes
: 

5convolutional10/kernel/Initializer/random_uniform/mulMul?convolutional10/kernel/Initializer/random_uniform/RandomUniform5convolutional10/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@convolutional10/kernel*'
_output_shapes
:

1convolutional10/kernel/Initializer/random_uniformAdd5convolutional10/kernel/Initializer/random_uniform/mul5convolutional10/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@convolutional10/kernel*'
_output_shapes
:
Į
convolutional10/kernel
VariableV2*
shape:*
shared_name *)
_class
loc:@convolutional10/kernel*
dtype0*
	container *'
_output_shapes
:
ø
convolutional10/kernel/AssignAssignconvolutional10/kernel1convolutional10/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@convolutional10/kernel*
validate_shape(*'
_output_shapes
:

convolutional10/kernel/readIdentityconvolutional10/kernel*
T0*)
_class
loc:@convolutional10/kernel*'
_output_shapes
:

&convolutional10/bias/Initializer/zerosConst*'
_class
loc:@convolutional10/bias*
valueB*    *
dtype0*
_output_shapes
:
Š
convolutional10/bias
VariableV2*
shape:*
shared_name *'
_class
loc:@convolutional10/bias*
dtype0*
	container *
_output_shapes
:
Ú
convolutional10/bias/AssignAssignconvolutional10/bias&convolutional10/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@convolutional10/bias*
validate_shape(*
_output_shapes
:

convolutional10/bias/readIdentityconvolutional10/bias*
T0*'
_class
loc:@convolutional10/bias*
_output_shapes
:
n
convolutional10/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional10/Conv2DConv2Dconvolutional9/Activationconvolutional10/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ļ
convolutional10/BiasAddBiasAddconvolutional10/Conv2Dconvolutional10/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
route1/concat_dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
o
route1/route1Identityconvolutional8/Activation*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ģ
7convolutional11/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@convolutional11/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ĩ
5convolutional11/kernel/Initializer/random_uniform/minConst*)
_class
loc:@convolutional11/kernel*
valueB
 *   ž*
dtype0*
_output_shapes
: 
Ĩ
5convolutional11/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@convolutional11/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 

?convolutional11/kernel/Initializer/random_uniform/RandomUniformRandomUniform7convolutional11/kernel/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@convolutional11/kernel*
dtype0*
seed2 *(
_output_shapes
:
ö
5convolutional11/kernel/Initializer/random_uniform/subSub5convolutional11/kernel/Initializer/random_uniform/max5convolutional11/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@convolutional11/kernel*
_output_shapes
: 

5convolutional11/kernel/Initializer/random_uniform/mulMul?convolutional11/kernel/Initializer/random_uniform/RandomUniform5convolutional11/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@convolutional11/kernel*(
_output_shapes
:

1convolutional11/kernel/Initializer/random_uniformAdd5convolutional11/kernel/Initializer/random_uniform/mul5convolutional11/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@convolutional11/kernel*(
_output_shapes
:
É
convolutional11/kernel
VariableV2*
shape:*
shared_name *)
_class
loc:@convolutional11/kernel*
dtype0*
	container *(
_output_shapes
:
ų
convolutional11/kernel/AssignAssignconvolutional11/kernel1convolutional11/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@convolutional11/kernel*
validate_shape(*(
_output_shapes
:

convolutional11/kernel/readIdentityconvolutional11/kernel*
T0*)
_class
loc:@convolutional11/kernel*(
_output_shapes
:
n
convolutional11/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional11/Conv2DConv2Droute1/route1convolutional11/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŗ
0convolutional11/BatchNorm/gamma/Initializer/onesConst*2
_class(
&$loc:@convolutional11/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
Á
convolutional11/BatchNorm/gamma
VariableV2*
shape:*
shared_name *2
_class(
&$loc:@convolutional11/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:

&convolutional11/BatchNorm/gamma/AssignAssignconvolutional11/BatchNorm/gamma0convolutional11/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*2
_class(
&$loc:@convolutional11/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:
Ģ
$convolutional11/BatchNorm/gamma/readIdentityconvolutional11/BatchNorm/gamma*
T0*2
_class(
&$loc:@convolutional11/BatchNorm/gamma*
_output_shapes	
:
˛
0convolutional11/BatchNorm/beta/Initializer/zerosConst*1
_class'
%#loc:@convolutional11/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes	
:
ŋ
convolutional11/BatchNorm/beta
VariableV2*
shape:*
shared_name *1
_class'
%#loc:@convolutional11/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:

%convolutional11/BatchNorm/beta/AssignAssignconvolutional11/BatchNorm/beta0convolutional11/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@convolutional11/BatchNorm/beta*
validate_shape(*
_output_shapes	
:
¨
#convolutional11/BatchNorm/beta/readIdentityconvolutional11/BatchNorm/beta*
T0*1
_class'
%#loc:@convolutional11/BatchNorm/beta*
_output_shapes	
:
Ā
7convolutional11/BatchNorm/moving_mean/Initializer/zerosConst*8
_class.
,*loc:@convolutional11/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
Í
%convolutional11/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *8
_class.
,*loc:@convolutional11/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:

,convolutional11/BatchNorm/moving_mean/AssignAssign%convolutional11/BatchNorm/moving_mean7convolutional11/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@convolutional11/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:
Ŋ
*convolutional11/BatchNorm/moving_mean/readIdentity%convolutional11/BatchNorm/moving_mean*
T0*8
_class.
,*loc:@convolutional11/BatchNorm/moving_mean*
_output_shapes	
:
Į
:convolutional11/BatchNorm/moving_variance/Initializer/onesConst*<
_class2
0.loc:@convolutional11/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
Õ
)convolutional11/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *<
_class2
0.loc:@convolutional11/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:
Ž
0convolutional11/BatchNorm/moving_variance/AssignAssign)convolutional11/BatchNorm/moving_variance:convolutional11/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*<
_class2
0.loc:@convolutional11/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:
É
.convolutional11/BatchNorm/moving_variance/readIdentity)convolutional11/BatchNorm/moving_variance*
T0*<
_class2
0.loc:@convolutional11/BatchNorm/moving_variance*
_output_shapes	
:

*convolutional11/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional11/Conv2D$convolutional11/BatchNorm/gamma/read#convolutional11/BatchNorm/beta/read*convolutional11/BatchNorm/moving_mean/read.convolutional11/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::
d
convolutional11/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional11/Activation	LeakyRelu*convolutional11/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
upsample1/sizeConst*
valueB"      *
dtype0*
_output_shapes
:
¸
	upsample1ResizeNearestNeighborconvolutional11/Activationupsample1/size*
align_corners( *
half_pixel_centers( *
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
route2/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

route2ConcatV2	upsample1convolutional5/Activationroute2/axis*

Tidx0*
T0*
N*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ģ
7convolutional12/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@convolutional12/kernel*%
valueB"           *
dtype0*
_output_shapes
:
Ĩ
5convolutional12/kernel/Initializer/random_uniform/minConst*)
_class
loc:@convolutional12/kernel*
valueB
 *Ĩ2Ŋ*
dtype0*
_output_shapes
: 
Ĩ
5convolutional12/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@convolutional12/kernel*
valueB
 *Ĩ2=*
dtype0*
_output_shapes
: 

?convolutional12/kernel/Initializer/random_uniform/RandomUniformRandomUniform7convolutional12/kernel/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@convolutional12/kernel*
dtype0*
seed2 *(
_output_shapes
:
ö
5convolutional12/kernel/Initializer/random_uniform/subSub5convolutional12/kernel/Initializer/random_uniform/max5convolutional12/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@convolutional12/kernel*
_output_shapes
: 

5convolutional12/kernel/Initializer/random_uniform/mulMul?convolutional12/kernel/Initializer/random_uniform/RandomUniform5convolutional12/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@convolutional12/kernel*(
_output_shapes
:

1convolutional12/kernel/Initializer/random_uniformAdd5convolutional12/kernel/Initializer/random_uniform/mul5convolutional12/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@convolutional12/kernel*(
_output_shapes
:
É
convolutional12/kernel
VariableV2*
shape:*
shared_name *)
_class
loc:@convolutional12/kernel*
dtype0*
	container *(
_output_shapes
:
ų
convolutional12/kernel/AssignAssignconvolutional12/kernel1convolutional12/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@convolutional12/kernel*
validate_shape(*(
_output_shapes
:

convolutional12/kernel/readIdentityconvolutional12/kernel*
T0*)
_class
loc:@convolutional12/kernel*(
_output_shapes
:
n
convolutional12/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional12/Conv2DConv2Droute2convolutional12/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŗ
0convolutional12/BatchNorm/gamma/Initializer/onesConst*2
_class(
&$loc:@convolutional12/BatchNorm/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
Á
convolutional12/BatchNorm/gamma
VariableV2*
shape:*
shared_name *2
_class(
&$loc:@convolutional12/BatchNorm/gamma*
dtype0*
	container *
_output_shapes	
:

&convolutional12/BatchNorm/gamma/AssignAssignconvolutional12/BatchNorm/gamma0convolutional12/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*2
_class(
&$loc:@convolutional12/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:
Ģ
$convolutional12/BatchNorm/gamma/readIdentityconvolutional12/BatchNorm/gamma*
T0*2
_class(
&$loc:@convolutional12/BatchNorm/gamma*
_output_shapes	
:
˛
0convolutional12/BatchNorm/beta/Initializer/zerosConst*1
_class'
%#loc:@convolutional12/BatchNorm/beta*
valueB*    *
dtype0*
_output_shapes	
:
ŋ
convolutional12/BatchNorm/beta
VariableV2*
shape:*
shared_name *1
_class'
%#loc:@convolutional12/BatchNorm/beta*
dtype0*
	container *
_output_shapes	
:

%convolutional12/BatchNorm/beta/AssignAssignconvolutional12/BatchNorm/beta0convolutional12/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*1
_class'
%#loc:@convolutional12/BatchNorm/beta*
validate_shape(*
_output_shapes	
:
¨
#convolutional12/BatchNorm/beta/readIdentityconvolutional12/BatchNorm/beta*
T0*1
_class'
%#loc:@convolutional12/BatchNorm/beta*
_output_shapes	
:
Ā
7convolutional12/BatchNorm/moving_mean/Initializer/zerosConst*8
_class.
,*loc:@convolutional12/BatchNorm/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
Í
%convolutional12/BatchNorm/moving_mean
VariableV2*
shape:*
shared_name *8
_class.
,*loc:@convolutional12/BatchNorm/moving_mean*
dtype0*
	container *
_output_shapes	
:

,convolutional12/BatchNorm/moving_mean/AssignAssign%convolutional12/BatchNorm/moving_mean7convolutional12/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@convolutional12/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:
Ŋ
*convolutional12/BatchNorm/moving_mean/readIdentity%convolutional12/BatchNorm/moving_mean*
T0*8
_class.
,*loc:@convolutional12/BatchNorm/moving_mean*
_output_shapes	
:
Į
:convolutional12/BatchNorm/moving_variance/Initializer/onesConst*<
_class2
0.loc:@convolutional12/BatchNorm/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
Õ
)convolutional12/BatchNorm/moving_variance
VariableV2*
shape:*
shared_name *<
_class2
0.loc:@convolutional12/BatchNorm/moving_variance*
dtype0*
	container *
_output_shapes	
:
Ž
0convolutional12/BatchNorm/moving_variance/AssignAssign)convolutional12/BatchNorm/moving_variance:convolutional12/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*<
_class2
0.loc:@convolutional12/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:
É
.convolutional12/BatchNorm/moving_variance/readIdentity)convolutional12/BatchNorm/moving_variance*
T0*<
_class2
0.loc:@convolutional12/BatchNorm/moving_variance*
_output_shapes	
:

*convolutional12/BatchNorm/FusedBatchNormV3FusedBatchNormV3convolutional12/Conv2D$convolutional12/BatchNorm/gamma/read#convolutional12/BatchNorm/beta/read*convolutional12/BatchNorm/moving_mean/read.convolutional12/BatchNorm/moving_variance/read*
epsilon%đ'7*
T0*
U0*
data_formatNHWC*
is_training( *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:::::
d
convolutional12/BatchNorm/ConstConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 

convolutional12/Activation	LeakyRelu*convolutional12/BatchNorm/FusedBatchNormV3*
T0*
alpha%ÍĖĖ=*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ģ
7convolutional13/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@convolutional13/kernel*%
valueB"            *
dtype0*
_output_shapes
:
Ĩ
5convolutional13/kernel/Initializer/random_uniform/minConst*)
_class
loc:@convolutional13/kernel*
valueB
 *2ĩž*
dtype0*
_output_shapes
: 
Ĩ
5convolutional13/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@convolutional13/kernel*
valueB
 *2ĩ>*
dtype0*
_output_shapes
: 

?convolutional13/kernel/Initializer/random_uniform/RandomUniformRandomUniform7convolutional13/kernel/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@convolutional13/kernel*
dtype0*
seed2 *'
_output_shapes
:
ö
5convolutional13/kernel/Initializer/random_uniform/subSub5convolutional13/kernel/Initializer/random_uniform/max5convolutional13/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@convolutional13/kernel*
_output_shapes
: 

5convolutional13/kernel/Initializer/random_uniform/mulMul?convolutional13/kernel/Initializer/random_uniform/RandomUniform5convolutional13/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@convolutional13/kernel*'
_output_shapes
:

1convolutional13/kernel/Initializer/random_uniformAdd5convolutional13/kernel/Initializer/random_uniform/mul5convolutional13/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@convolutional13/kernel*'
_output_shapes
:
Į
convolutional13/kernel
VariableV2*
shape:*
shared_name *)
_class
loc:@convolutional13/kernel*
dtype0*
	container *'
_output_shapes
:
ø
convolutional13/kernel/AssignAssignconvolutional13/kernel1convolutional13/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@convolutional13/kernel*
validate_shape(*'
_output_shapes
:

convolutional13/kernel/readIdentityconvolutional13/kernel*
T0*)
_class
loc:@convolutional13/kernel*'
_output_shapes
:

&convolutional13/bias/Initializer/zerosConst*'
_class
loc:@convolutional13/bias*
valueB*    *
dtype0*
_output_shapes
:
Š
convolutional13/bias
VariableV2*
shape:*
shared_name *'
_class
loc:@convolutional13/bias*
dtype0*
	container *
_output_shapes
:
Ú
convolutional13/bias/AssignAssignconvolutional13/bias&convolutional13/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@convolutional13/bias*
validate_shape(*
_output_shapes
:

convolutional13/bias/readIdentityconvolutional13/bias*
T0*'
_class
loc:@convolutional13/bias*
_output_shapes
:
n
convolutional13/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

convolutional13/Conv2DConv2Dconvolutional12/Activationconvolutional13/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ļ
convolutional13/BiasAddBiasAddconvolutional13/Conv2Dconvolutional13/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙"