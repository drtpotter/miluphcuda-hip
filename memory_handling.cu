/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com
 *
 * @section     LICENSE
 * Copyright (c) 2019 Christoph Schaefer
 *
 * This file is part of miluphcuda.
 *
 * miluphcuda is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * miluphcuda is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "miluph.h"
#include "memory_handling.h"
#include "aneos.h"


/* allocate memory on the device for pointmasses */
int allocate_pointmass_memory(struct Pointmass *a, int allocate_immutables)
{
    int rc = 0;

	cudaVerify(hipMalloc((void**)&a->x, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->vx, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->ax, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->feedback_ax, memorySizeForPointmasses));
#if DIM > 1
	cudaVerify(hipMalloc((void**)&a->y, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->vy, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->ay, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->feedback_ay, memorySizeForPointmasses));
# if DIM > 2
	cudaVerify(hipMalloc((void**)&a->z, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->vz, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->az, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->feedback_az, memorySizeForPointmasses));
# endif
#endif
	cudaVerify(hipMalloc((void**)&a->m, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->rmin, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->rmax, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&a->feels_particles, integermemorySizeForPointmasses));

    return rc;
}



/* allocate memory on the device for particles */
int allocate_particles_memory(struct Particle *a, int allocate_immutables)
{
    int rc = 0;

#if TENSORIAL_CORRECTION
    // also moved to p_device only
//	cudaVerify(hipMalloc((void**)&a->tensorialCorrectionMatrix, memorySizeForStress));
    // not needed anymore, let's save memory --- tschakka!
/*    if (allocate_immutables) {
        cudaVerify(hipMalloc((void**)&a->tensorialCorrectiondWdrr, MAX_NUM_INTERACTIONS * maxNumberOfParticles * sizeof(double)));
    } */
#endif

#if INTEGRATE_ENERGY
	cudaVerify(hipMalloc((void**)&a->dedt, memorySizeForParticles));
#endif

#if ARTIFICIAL_VISCOSITY
	cudaVerify(hipMalloc((void**)&a->muijmax, memorySizeForParticles));
#endif

	cudaVerify(hipMalloc((void**)&a->drhodt, memorySizeForParticles));

#if SOLID
	cudaVerify(hipMalloc((void**)&a->S, memorySizeForStress));
	cudaVerify(hipMalloc((void**)&a->dSdt, memorySizeForStress));
	cudaVerify(hipMalloc((void**)&a->local_strain, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->ep, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->edotp, memorySizeForParticles));
#endif

#if NAVIER_STOKES
	cudaVerify(hipMalloc((void**)&a->Tshear, memorySizeForStress));
#endif

#if INVISCID_SPH
	cudaVerify(hipMalloc((void**)&a->beta, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->beta_old, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->divv_old, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->dbetadt, memorySizeForParticles));
#endif

#if FRAGMENTATION
	memorySizeForActivationThreshold = maxNumberOfParticles * MAX_NUM_FLAWS * sizeof(double);
	cudaVerify(hipMalloc((void**)&a->d, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->damage_total, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->dddt, memorySizeForParticles));

	cudaVerify(hipMalloc((void**)&a->numFlaws, memorySizeForInteractions));
	cudaVerify(hipMalloc((void**)&a->numActiveFlaws, memorySizeForInteractions));
    if (allocate_immutables) {
	    cudaVerify(hipMalloc((void**)&a->flaws, memorySizeForActivationThreshold));
    }
# if PALPHA_POROSITY
	cudaVerify(hipMalloc((void**)&a->damage_porjutzi, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->ddamage_porjutzidt, memorySizeForParticles));
# endif
#endif

    if (allocate_immutables) {
        cudaVerify(hipMalloc((void**)&a->h0, memorySizeForParticles));
    }

#if GHOST_BOUNDARIES
	cudaVerify(hipMalloc((void**)&a->real_partner, memorySizeForInteractions));
#endif

#if PALPHA_POROSITY
	cudaVerify(hipMalloc((void**)&a->pold, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->alpha_jutzi, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->alpha_jutzi_old, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->dalphadt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->dp, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->dalphadp, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->dalphadrho, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->delpdelrho, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->delpdele, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->f, memorySizeForParticles));
#endif

#if SIRONO_POROSITY
    cudaVerify(hipMalloc((void**)&a->compressive_strength, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->tensile_strength, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->shear_strength, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->K, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->rho_0prime, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->rho_c_plus, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->rho_c_minus, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->flag_rho_0prime, memorySizeForInteractions));
    cudaVerify(hipMalloc((void**)&a->flag_plastic, memorySizeForInteractions));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(hipMalloc((void**)&a->alpha_epspor, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->dalpha_epspordt, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->epsilon_v, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->depsilon_vdt, memorySizeForParticles));
#endif

    cudaVerify(hipMalloc((void**)&a->x0, memorySizeForTree));
#if DIM > 1
    cudaVerify(hipMalloc((void**)&a->y0, memorySizeForTree));
#if DIM > 2
    cudaVerify(hipMalloc((void**)&a->z0, memorySizeForTree));
#endif
#endif
	cudaVerify(hipMalloc((void**)&a->x, memorySizeForTree));
#if DIM > 1
	cudaVerify(hipMalloc((void**)&a->y, memorySizeForTree));
#endif
	cudaVerify(hipMalloc((void**)&a->vx, memorySizeForParticles));
#if DIM > 1
	cudaVerify(hipMalloc((void**)&a->vy, memorySizeForParticles));
#endif
	cudaVerify(hipMalloc((void**)&a->dxdt, memorySizeForParticles));
#if DIM > 1
 	cudaVerify(hipMalloc((void**)&a->dydt, memorySizeForParticles));
#endif

#if XSPH
	cudaVerify(hipMalloc((void**)&a->xsphvx, memorySizeForParticles));
#if DIM > 1
	cudaVerify(hipMalloc((void**)&a->xsphvy, memorySizeForParticles));
#endif
#endif
	cudaVerify(hipMalloc((void**)&a->ax, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->g_ax, memorySizeForParticles));
#if DIM > 1
	cudaVerify(hipMalloc((void**)&a->ay, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->g_ay, memorySizeForParticles));
#endif
	cudaVerify(hipMalloc((void**)&a->m, memorySizeForTree));
	cudaVerify(hipMalloc((void**)&a->h, memorySizeForParticles));
#if INTEGRATE_SML
	cudaVerify(hipMalloc((void**)&a->dhdt, memorySizeForParticles));
#endif

#if SML_CORRECTION
	cudaVerify(hipMalloc((void**)&a->sml_omega, memorySizeForParticles));
#endif

	cudaVerify(hipMalloc((void**)&a->rho, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->p, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->e, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->cs, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->noi, memorySizeForInteractions));
	cudaVerify(hipMalloc((void**)&a->depth, memorySizeForInteractions));
#if MORE_OUTPUT
	cudaVerify(hipMalloc((void**)&a->p_min, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->p_max, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->rho_min, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->rho_max, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->e_min, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->e_max, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->cs_min, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&a->cs_max, memorySizeForParticles));
#endif
// moved to p_device only, so we don't need mem here anymore
//	cudaVerify(hipMalloc((void**)&a->materialId, memorySizeForInteractions));

#if JC_PLASTICITY
	cudaVerify(hipMalloc((void**)&a->T, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->dTdt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->jc_f, memorySizeForParticles));
#endif

#if DIM > 2
	cudaVerify(hipMalloc((void**)&a->z, memorySizeForTree));
	cudaVerify(hipMalloc((void**)&a->dzdt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->vz, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->az, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&a->g_az, memorySizeForParticles));
#if XSPH
	cudaVerify(hipMalloc((void**)&a->xsphvz, memorySizeForParticles));
#endif
#endif
	cudaVerify(hipMemset(a->ax, 0, memorySizeForParticles));
	cudaVerify(hipMemset(a->g_ax, 0, memorySizeForParticles));
#if DIM > 1
	cudaVerify(hipMemset(a->ay, 0, memorySizeForParticles));
	cudaVerify(hipMemset(a->g_ay, 0, memorySizeForParticles));
#if DIM == 3
	cudaVerify(hipMemset(a->az, 0, memorySizeForParticles));
	cudaVerify(hipMemset(a->g_az, 0, memorySizeForParticles));
#endif
#endif

    return rc;
}



int copy_gravitational_accels_device_to_device(struct Particle *dst, struct Particle *src)
{
    int rc = 0;
    cudaVerify(hipMemcpy(dst->g_ax, src->g_ax, memorySizeForParticles, hipMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(hipMemcpy(dst->g_ay, src->g_ay, memorySizeForParticles, hipMemcpyDeviceToDevice));
#if DIM > 2
    cudaVerify(hipMemcpy(dst->g_az, src->g_az, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif
#endif

    return rc;
}



int copy_pointmass_derivatives_device_to_device(struct Pointmass *dst, struct Pointmass *src)
{
    int rc = 0;
    cudaVerify(hipMemcpy(dst->ax, src->ax, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->vx, src->vx, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->feedback_ax, src->feedback_ax, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(hipMemcpy(dst->ay, src->ay, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->vy, src->vy, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->feedback_ay, src->feedback_ay, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
# if DIM > 2
    cudaVerify(hipMemcpy(dst->az, src->az, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->vz, src->vz, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->feedback_az, src->feedback_az, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
# endif
#endif

    return rc;
}



int copy_particles_derivatives_device_to_device(struct Particle *dst, struct Particle *src)
{
    int rc = 0;

    cudaVerify(hipMemcpy(dst->ax, src->ax, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->g_ax, src->g_ax, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->dxdt, src->dxdt, memorySizeForParticles, hipMemcpyDeviceToDevice));

#if DIM > 1
    cudaVerify(hipMemcpy(dst->ay, src->ay, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->g_ay, src->g_ay, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->dydt, src->dydt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#if DIM > 2
    cudaVerify(hipMemcpy(dst->az, src->az, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->g_az, src->g_az, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->dzdt, src->dzdt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif
#endif

    cudaVerify(hipMemcpy(dst->drhodt, src->drhodt, memorySizeForParticles, hipMemcpyDeviceToDevice));

#if INTEGRATE_SML
    cudaVerify(hipMemcpy(dst->dhdt, src->dhdt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if SML_CORRECTION
    cudaVerify(hipMemcpy(dst->sml_omega, src->sml_omega, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if PALPHA_POROSITY
    cudaVerify(hipMemcpy(dst->dalphadt, src->dalphadt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#if FRAGMENTATION
    cudaVerify(hipMemcpy(dst->ddamage_porjutzidt, src->ddamage_porjutzidt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif
#endif

#if EPSALPHA_POROSITY
    cudaVerify(hipMemcpy(dst->dalpha_epspordt, src->dalpha_epspordt, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->depsilon_vdt, src->depsilon_vdt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if INTEGRATE_ENERGY
    cudaVerify(hipMemcpy(dst->dedt, src->dedt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if SOLID
    cudaVerify(hipMemcpy(dst->dSdt, src->dSdt, memorySizeForStress, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->edotp, src->edotp, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if INVISCID_SPH
	cudaVerify(hipMemcpy(dst->dbetadt, src->dbetadt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if JC_PLASTICITY
    cudaVerify(hipMemcpy(dst->dTdt, src->dTdt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if FRAGMENTATION
    cudaVerify(hipMemcpy(dst->dddt, src->dddt, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->numActiveFlaws, src->numActiveFlaws, memorySizeForInteractions, hipMemcpyDeviceToDevice));
#endif

    return rc;
}



int copy_pointmass_immutables_device_to_device(struct Pointmass *dst, struct Pointmass *src)
{
    int rc = 0;

    cudaVerify(hipMemcpy((*dst).m, (*src).m, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy((*dst).feels_particles, (*src).feels_particles, integermemorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy((*dst).rmin, (*src).rmin, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy((*dst).rmax, (*src).rmax, memorySizeForPointmasses, hipMemcpyDeviceToDevice));

    return rc;
}



int copy_particles_immutables_device_to_device(struct Particle *dst, struct Particle *src)
{
    int rc = 0;

    cudaVerify(hipMemcpy((*dst).x0, (*src).x0, memorySizeForTree, hipMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(hipMemcpy((*dst).y0, (*src).y0, memorySizeForTree, hipMemcpyDeviceToDevice));
#endif
#if DIM > 2
    cudaVerify(hipMemcpy((*dst).z0, (*src).z0, memorySizeForTree, hipMemcpyDeviceToDevice));
#endif
    cudaVerify(hipMemcpy((*dst).m, (*src).m, memorySizeForTree, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy((*dst).h, (*src).h, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy((*dst).cs, (*src).cs, memorySizeForParticles, hipMemcpyDeviceToDevice));
    //cudaVerify(hipMemcpy((*dst).materialId, (*src).materialId, memorySizeForInteractions, hipMemcpyDeviceToDevice));
#if FRAGMENTATION
	cudaVerify(hipMemcpy(dst->numFlaws, src->numFlaws, memorySizeForInteractions, hipMemcpyDeviceToDevice));
    //cudaVerify(hipMemcpy(dst->flaws, src->flaws, memorySizeForActivationThreshold, hipMemcpyDeviceToDevice));
#endif

    return rc;
}



int copy_pointmass_variables_device_to_device(struct Pointmass *dst, struct Pointmass *src)
{
    int rc = 0;
    cudaVerify(hipMemcpy(dst->x, src->x, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    // mass is variable
    cudaVerify(hipMemcpy(dst->m, src->m, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->vx, src->vx, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(hipMemcpy(dst->y, src->y, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->vy, src->vy, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
# if DIM > 2
    cudaVerify(hipMemcpy(dst->z, src->z, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->vz, src->vz, memorySizeForPointmasses, hipMemcpyDeviceToDevice));
# endif
#endif

    return rc;
}



int copy_particles_variables_device_to_device(struct Particle *dst, struct Particle *src)
{
    int rc = 0;

    cudaVerify(hipMemcpy(dst->x, src->x, memorySizeForTree, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->x0, src->x0, memorySizeForTree, hipMemcpyDeviceToDevice));
    // materialId moved to p_device aka p_rhs only
    //cudaVerify(hipMemcpy((*dst).materialId, (*src).materialId, memorySizeForInteractions, hipMemcpyDeviceToDevice));
#if DIM > 1
    cudaVerify(hipMemcpy(dst->y, src->y, memorySizeForTree, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->y0, src->y0, memorySizeForTree, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->vy, src->vy, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif
#if DIM > 2
    cudaVerify(hipMemcpy(dst->z0, src->z0, memorySizeForTree, hipMemcpyDeviceToDevice));
#endif

    cudaVerify(hipMemcpy(dst->vx, src->vx, memorySizeForParticles, hipMemcpyDeviceToDevice));

    cudaVerify(hipMemcpy(dst->rho, src->rho, memorySizeForParticles, hipMemcpyDeviceToDevice));

    cudaVerify(hipMemcpy(dst->h, src->h, memorySizeForParticles, hipMemcpyDeviceToDevice));

#if INTEGRATE_ENERGY
    cudaVerify(hipMemcpy(dst->e, src->e, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if PALPHA_POROSITY
    cudaVerify(hipMemcpy(dst->alpha_jutzi, src->alpha_jutzi, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->alpha_jutzi_old, src->alpha_jutzi, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->dalphadp, src->dalphadp, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->dalphadrho, src->dalphadrho, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->dp, src->dp, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->delpdelrho, src->delpdelrho, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->delpdele, src->delpdele, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->f, src->f, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->p, src->p, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->pold, src->pold, memorySizeForParticles, hipMemcpyDeviceToDevice));
# if FRAGMENTATION
    cudaVerify(hipMemcpy(dst->damage_porjutzi, src->damage_porjutzi, memorySizeForParticles, hipMemcpyDeviceToDevice));
# endif
#endif

#if MORE_OUTPUT
    cudaVerify(hipMemcpy(dst->p_min, src->p_min, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->p_max, src->p_max, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->rho_min, src->rho_min, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->rho_max, src->rho_max, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->e_min, src->e_min, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->e_max, src->e_max, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->cs_min, src->cs_min, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->cs_max, src->cs_max, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if SIRONO_POROSITY
    cudaVerify(hipMemcpy(dst->compressive_strength, src->compressive_strength, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->tensile_strength, src->tensile_strength, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->shear_strength, src->shear_strength, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->K, src->K, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->rho_0prime, src->rho_0prime, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->rho_c_plus, src->rho_c_plus, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->rho_c_minus, src->rho_c_minus, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->flag_rho_0prime, src->flag_rho_0prime, memorySizeForInteractions, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->flag_plastic, src->flag_plastic, memorySizeForInteractions, hipMemcpyDeviceToDevice));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(hipMemcpy(dst->alpha_epspor, src->alpha_epspor, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->dalpha_epspordt, src->dalpha_epspordt, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->epsilon_v, src->epsilon_v, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->depsilon_vdt, src->depsilon_vdt, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if DIM > 2
    cudaVerify(hipMemcpy(dst->z, src->z, memorySizeForTree, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->vz, src->vz, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif
#if SOLID
    cudaVerify(hipMemcpy(dst->S, src->S, memorySizeForStress, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->ep, src->ep, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif
#if NAVIER_STOKES
    cudaVerify(hipMemcpy(dst->Tshear, src->Tshear, memorySizeForStress, hipMemcpyDeviceToDevice));
#endif

#if INVISCID_SPH
    cudaVerify(hipMemcpy(dst->beta, src->beta, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->beta_old, src->beta_old, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->divv_old, src->divv_old, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if JC_PLASTICITY
    cudaVerify(hipMemcpy(dst->T, src->T, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->jc_f, src->jc_f, memorySizeForParticles, hipMemcpyDeviceToDevice));
#endif

#if FRAGMENTATION
    cudaVerify(hipMemcpy(dst->d, src->d, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->damage_total, src->damage_total, memorySizeForParticles, hipMemcpyDeviceToDevice));
    cudaVerify(hipMemcpy(dst->numActiveFlaws, src->numActiveFlaws, memorySizeForInteractions, hipMemcpyDeviceToDevice));
#endif

    return rc;
}



/* free runge-kutta memory for pointmasses on the device */
int free_pointmass_memory(struct Pointmass *a, int free_immutables)
{
    int rc = 0;
	cudaVerify(hipFree(a->x));
	cudaVerify(hipFree(a->vx));
	cudaVerify(hipFree(a->ax));
	cudaVerify(hipFree(a->feedback_ax));
	cudaVerify(hipFree(a->m));
	cudaVerify(hipFree(a->feels_particles));
	cudaVerify(hipFree(a->rmin));
	cudaVerify(hipFree(a->rmax));
#if DIM > 1
	cudaVerify(hipFree(a->y));
	cudaVerify(hipFree(a->vy));
	cudaVerify(hipFree(a->ay));
	cudaVerify(hipFree(a->feedback_ay));
# if DIM > 2
	cudaVerify(hipFree(a->z));
	cudaVerify(hipFree(a->vz));
	cudaVerify(hipFree(a->az));
	cudaVerify(hipFree(a->feedback_az));
# endif
#endif

    return rc;
}



/* free runge-kutta memory on the device */
int free_particles_memory(struct Particle *a, int free_immutables)
{
    int rc = 0;

	cudaVerify(hipFree(a->x));
	cudaVerify(hipFree(a->x0));
	cudaVerify(hipFree(a->dxdt));
	cudaVerify(hipFree(a->vx));
	cudaVerify(hipFree(a->ax));
	cudaVerify(hipFree(a->g_ax));
	cudaVerify(hipFree(a->m));
#if DIM > 1
	cudaVerify(hipFree(a->dydt));
	cudaVerify(hipFree(a->y));
	cudaVerify(hipFree(a->y0));
	cudaVerify(hipFree(a->vy0));
	cudaVerify(hipFree(a->vy));
	cudaVerify(hipFree(a->ay));
	cudaVerify(hipFree(a->g_ay));
#endif

#if XSPH
	cudaVerify(hipFree(a->xsphvx));
#if DIM > 1
	cudaVerify(hipFree(a->xsphvy));
#endif
#endif
	cudaVerify(hipFree(a->h));
	cudaVerify(hipFree(a->rho));
	cudaVerify(hipFree(a->p));
	cudaVerify(hipFree(a->e));
	cudaVerify(hipFree(a->cs));
	cudaVerify(hipFree(a->noi));
	cudaVerify(hipFree(a->depth));
#if MORE_OUTPUT
	cudaVerify(hipFree(a->p_min));
	cudaVerify(hipFree(a->p_max));
	cudaVerify(hipFree(a->rho_min));
	cudaVerify(hipFree(a->rho_max));
	cudaVerify(hipFree(a->e_min));
	cudaVerify(hipFree(a->e_max));
	cudaVerify(hipFree(a->cs_min));
	cudaVerify(hipFree(a->cs_max));
#endif
    // materialId only on p_device
	//cudaVerify(hipFree(a->materialId));
#if DIM > 2
	cudaVerify(hipFree(a->z));
	cudaVerify(hipFree(a->z0));
	cudaVerify(hipFree(a->dzdt));
	cudaVerify(hipFree(a->vz));
#if XSPH
	cudaVerify(hipFree(a->xsphvz));
#endif
	cudaVerify(hipFree(a->az));
	cudaVerify(hipFree(a->g_az));
#endif


#if ARTIFICIAL_VISCOSITY
	cudaVerify(hipFree(a->muijmax));
#endif
#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)
	cudaVerify(hipFree(a->divv));
	cudaVerify(hipFree(a->curlv));
#endif

#if INVISCID_SPH
	cudaVerify(hipFree(a->beta));
	cudaVerify(hipFree(a->beta_old));
	cudaVerify(hipFree(a->divv_old));
	cudaVerify(hipFree(a->dbetadt));
#endif

#if TENSORIAL_CORRECTION
	//cudaVerify(hipFree(a->tensorialCorrectionMatrix));
    /*
    if (free_immutables) {
	    cudaVerify(hipFree(a->tensorialCorrectiondWdrr));
    } */
#endif

#if INTEGRATE_ENERGY
	cudaVerify(hipFree(a->dedt));
#endif
#if GHOST_BOUNDARIES
	cudaVerify(hipFree(a->real_partner));
#endif

	cudaVerify(hipFree(a->drhodt));

#if INTEGRATE_SML
	cudaVerify(hipFree(a->dhdt));
#endif

#if SML_CORRECTION
    cudaVerify(hipFree(a->sml_omega));
#endif

#if SOLID
	cudaVerify(hipFree(a->S));
	cudaVerify(hipFree(a->dSdt));
	cudaVerify(hipFree(a->local_strain));
    cudaVerify(hipFree(a->ep));
    cudaVerify(hipFree(a->edotp));
#endif
#if NAVIER_STOKES
	cudaVerify(hipFree(a->Tshear));
#endif

#if JC_PLASTICITY
	cudaVerify(hipFree(a->T));
	cudaVerify(hipFree(a->dTdt));
	cudaVerify(hipFree(a->jc_f));
#endif

#if PALPHA_POROSITY
	cudaVerify(hipFree(a->pold));
	cudaVerify(hipFree(a->alpha_jutzi));
	cudaVerify(hipFree(a->alpha_jutzi_old));
	cudaVerify(hipFree(a->dalphadt));
	cudaVerify(hipFree(a->f));
	cudaVerify(hipFree(a->dalphadp));
	cudaVerify(hipFree(a->dp));
	cudaVerify(hipFree(a->delpdelrho));
	cudaVerify(hipFree(a->delpdele));
	cudaVerify(hipFree(a->dalphadrho));
#endif

#if SIRONO_POROSITY
    cudaVerify(hipFree(a->compressive_strength));
    cudaVerify(hipFree(a->tensile_strength));
    cudaVerify(hipFree(a->shear_strength));
    cudaVerify(hipFree(a->K));
    cudaVerify(hipFree(a->rho_0prime));
    cudaVerify(hipFree(a->rho_c_plus));
    cudaVerify(hipFree(a->rho_c_minus));
    cudaVerify(hipFree(a->flag_rho_0prime));
    cudaVerify(hipFree(a->flag_plastic));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(hipFree(a->alpha_epspor));
    cudaVerify(hipFree(a->dalpha_epspordt));
    cudaVerify(hipFree(a->epsilon_v));
    cudaVerify(hipFree(a->depsilon_vdt));
#endif

#if FRAGMENTATION
	cudaVerify(hipFree(a->d));
	cudaVerify(hipFree(a->damage_total));
	cudaVerify(hipFree(a->dddt));
	cudaVerify(hipFree(a->numFlaws));
	cudaVerify(hipFree(a->numActiveFlaws));
    if (free_immutables) {
	    cudaVerify(hipFree(a->flaws));
    }
    if (free_immutables) {
	    cudaVerify(hipFree(a->h0));
    }
# if PALPHA_POROSITY
	cudaVerify(hipFree(a->damage_porjutzi));
	cudaVerify(hipFree(a->ddamage_porjutzidt));
# endif
#endif

    return rc;
}



/* allocate memory for tree and basic particle struct */
int init_allocate_memory(void)
{
    int rc = 0;

	numberOfNodes = ceil(2.5 * maxNumberOfParticles);
    if (numberOfNodes < 1024*numberOfMultiprocessors)
        numberOfNodes = 1024*numberOfMultiprocessors;

#define WARPSIZE 32
    
    while ((numberOfNodes & (WARPSIZE-1)) != 0)
        numberOfNodes++;

	if (param.verbose) {
        fprintf(stdout, "\nAllocating memory for %d particles...\n", numberOfParticles);
	    fprintf(stdout, "Allocating memory for %d pointmasses...\n", numberOfPointmasses);
        fprintf(stdout, "Number of nodes of tree: %d\n", numberOfNodes);
    }

	memorySizeForParticles = maxNumberOfParticles * sizeof(double);
	memorySizeForPointmasses = numberOfPointmasses * sizeof(double);
	integermemorySizeForPointmasses = numberOfPointmasses * sizeof(int);
	memorySizeForTree = numberOfNodes * sizeof(double);
	memorySizeForStress = maxNumberOfParticles * DIM * DIM * sizeof(double);
	memorySizeForChildren = numberOfChildren * (numberOfNodes-numberOfRealParticles) * sizeof(int);
	memorySizeForInteractions = maxNumberOfParticles * sizeof(int);

    cudaVerify(hipHostMalloc((void**)&p_host.x, memorySizeForTree));
	cudaVerify(hipHostMalloc((void**)&p_host.vx, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.ax, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.g_ax, memorySizeForParticles));
#if DIM > 1
    cudaVerify(hipHostMalloc((void**)&p_host.y, memorySizeForTree));
	cudaVerify(hipHostMalloc((void**)&p_host.vy, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.ay, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.g_ay, memorySizeForParticles));
#endif
#if DIM > 2
    cudaVerify(hipHostMalloc((void**)&p_host.z, memorySizeForTree));
    cudaVerify(hipHostMalloc((void**)&p_host.vz, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.az, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.g_az, memorySizeForParticles));
#endif
    cudaVerify(hipHostMalloc((void**)&p_host.m, memorySizeForTree));
    cudaVerify(hipHostMalloc((void**)&p_host.h, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.rho, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.p, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.e, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.cs, memorySizeForParticles));

#if GRAVITATING_POINT_MASSES
	cudaVerify(hipHostMalloc((void**)&pointmass_host.x, memorySizeForPointmasses));
	cudaVerify(hipHostMalloc((void**)&pointmass_host.vx, memorySizeForPointmasses));
	cudaVerify(hipHostMalloc((void**)&pointmass_host.ax, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.x, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.vx, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.ax, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.feedback_ax, memorySizeForPointmasses));
#if DIM > 1
	cudaVerify(hipHostMalloc((void**)&pointmass_host.y, memorySizeForPointmasses));
	cudaVerify(hipHostMalloc((void**)&pointmass_host.vy, memorySizeForPointmasses));
	cudaVerify(hipHostMalloc((void**)&pointmass_host.ay, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.y, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.vy, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.ay, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.feedback_ay, memorySizeForPointmasses));
#if DIM > 2
	cudaVerify(hipHostMalloc((void**)&pointmass_host.z, memorySizeForPointmasses));
	cudaVerify(hipHostMalloc((void**)&pointmass_host.vz, memorySizeForPointmasses));
	cudaVerify(hipHostMalloc((void**)&pointmass_host.az, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.z, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.vz, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.az, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.feedback_az, memorySizeForPointmasses));
#endif
#endif
	cudaVerify(hipHostMalloc((void**)&pointmass_host.rmin, memorySizeForPointmasses));
	cudaVerify(hipHostMalloc((void**)&pointmass_host.rmax, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.rmin, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.rmax, memorySizeForPointmasses));
	cudaVerify(hipHostMalloc((void**)&pointmass_host.m, memorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.m, memorySizeForPointmasses));
	cudaVerify(hipHostMalloc((void**)&pointmass_host.feels_particles, integermemorySizeForPointmasses));
	cudaVerify(hipMalloc((void**)&pointmass_device.feels_particles, integermemorySizeForPointmasses));
#endif

#if MORE_OUTPUT
	cudaVerify(hipHostMalloc((void**)&p_host.p_min, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.p_max, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.rho_min, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.rho_max, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.e_min, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.e_max, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.cs_min, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.cs_max, memorySizeForParticles));
#endif

	cudaVerify(hipHostMalloc((void**)&p_host.noi, memorySizeForInteractions));
	cudaVerify(hipHostMalloc((void**)&p_host.depth, memorySizeForInteractions));
	cudaVerify(hipHostMalloc((void**)&interactions_host, memorySizeForInteractions*MAX_NUM_INTERACTIONS));
	cudaVerify(hipHostMalloc((void**)&p_host.materialId, memorySizeForInteractions));
	cudaVerify(hipHostMalloc((void**)&childList_host, memorySizeForChildren));

#if ARTIFICIAL_VISCOSITY
	cudaVerify(hipMalloc((void**)&p_device.muijmax, memorySizeForParticles));
#endif

#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)
	cudaVerify(hipMalloc((void**)&p_device.divv, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.curlv, memorySizeForParticles*DIM));
#endif

#if INVISCID_SPH
	cudaVerify(hipMalloc((void**)&p_device.beta, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.beta_old, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.divv_old, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dbetadt, memorySizeForParticles));
#endif

#if TENSORIAL_CORRECTION
	cudaVerify(hipMalloc((void**)&p_device.tensorialCorrectionMatrix, memorySizeForStress));
	//cudaVerify(hipMalloc((void**)&p_device.tensorialCorrectiondWdrr, MAX_NUM_INTERACTIONS * maxNumberOfParticles * sizeof(double)));
#endif

#if SHEPARD_CORRECTION
	cudaVerify(hipMalloc((void**)&p_device.shepard_correction, memorySizeForParticles));
#endif

#if INTEGRATE_ENERGY
	cudaVerify(hipHostMalloc((void**)&p_host.dedt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dedt, memorySizeForParticles));
#endif

	cudaVerify(hipHostMalloc((void**)&p_host.drhodt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.drhodt, memorySizeForParticles));

#if SOLID
	cudaVerify(hipHostMalloc((void**)&p_host.S, memorySizeForStress));
	cudaVerify(hipHostMalloc((void**)&p_host.dSdt, memorySizeForStress));
	cudaVerify(hipMalloc((void**)&p_device.S, memorySizeForStress));
	cudaVerify(hipMalloc((void**)&p_device.dSdt, memorySizeForStress));
	cudaVerify(hipHostMalloc((void**)&p_host.local_strain, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.local_strain, memorySizeForParticles));
	cudaVerify(hipMalloc((void**) &p_device.sigma, memorySizeForStress));
    cudaVerify(hipMalloc((void**)&p_device.plastic_f, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.ep, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.ep, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.edotp, memorySizeForParticles));
#endif

#if NAVIER_STOKES
	cudaVerify(hipHostMalloc((void**)&p_host.Tshear, memorySizeForStress));
	cudaVerify(hipMalloc((void**)&p_device.Tshear, memorySizeForStress));
	cudaVerify(hipMalloc((void**)&p_device.eta, memorySizeForParticles));
#endif

#if ARTIFICIAL_STRESS
	cudaVerify(hipMalloc((void**) &p_device.R, memorySizeForStress));
#endif

#if JC_PLASTICITY
	cudaVerify(hipMalloc((void**)&p_device.T, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.T, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dTdt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.jc_f, memorySizeForParticles));
#endif

#if FRAGMENTATION
	memorySizeForActivationThreshold = maxNumberOfParticles * MAX_NUM_FLAWS * sizeof(double);
	cudaVerify(hipHostMalloc((void**)&p_host.d, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.dddt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.d, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.damage_total, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dddt, memorySizeForParticles));

	cudaVerify(hipHostMalloc((void**)&p_host.numFlaws, memorySizeForInteractions));
	cudaVerify(hipMalloc((void**)&p_device.numFlaws, memorySizeForInteractions));
	cudaVerify(hipHostMalloc((void**)&p_host.numActiveFlaws, memorySizeForInteractions));
	cudaVerify(hipMalloc((void**)&p_device.numActiveFlaws, memorySizeForInteractions));
	cudaVerify(hipHostMalloc((void**)&p_host.flaws, memorySizeForActivationThreshold));
	cudaVerify(hipMalloc((void**)&p_device.flaws, memorySizeForActivationThreshold));
# if PALPHA_POROSITY
    cudaVerify(hipHostMalloc((void**)&p_host.damage_porjutzi, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.ddamage_porjutzidt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.damage_porjutzi, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.ddamage_porjutzidt, memorySizeForParticles));
# endif
#endif

	cudaVerify(hipMalloc((void**)&p_device.h0, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.h0, memorySizeForParticles));

#if GHOST_BOUNDARIES
	cudaVerify(hipMalloc((void**)&p_device.real_partner, memorySizeForInteractions));
#endif

#if PALPHA_POROSITY
	cudaVerify(hipHostMalloc((void**)&p_host.alpha_jutzi, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.alpha_jutzi_old, memorySizeForParticles));
	cudaVerify(hipHostMalloc((void**)&p_host.pold, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.dalphadt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.pold, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.alpha_jutzi, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.alpha_jutzi_old, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dalphadt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dalphadp, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dp, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dalphadrho, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.f, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.delpdelrho, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.delpdele, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.cs_old, memorySizeForParticles));
#endif

#if SIRONO_POROSITY
    cudaVerify(hipHostMalloc((void**)&p_host.compressive_strength, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.tensile_strength, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.shear_strength, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.rho_0prime, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.rho_c_plus, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.rho_c_minus, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.K, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.flag_rho_0prime, memorySizeForInteractions));
    cudaVerify(hipHostMalloc((void**)&p_host.flag_plastic, memorySizeForInteractions));
    cudaVerify(hipMalloc((void**)&p_device.compressive_strength, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.tensile_strength, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.shear_strength, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.K, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.rho_0prime, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.rho_c_plus, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.rho_c_minus, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.flag_rho_0prime, memorySizeForInteractions));
    cudaVerify(hipMalloc((void**)&p_device.flag_plastic, memorySizeForInteractions));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(hipHostMalloc((void**)&p_host.alpha_epspor, memorySizeForParticles));
    cudaVerify(hipHostMalloc((void**)&p_host.epsilon_v, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.alpha_epspor, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.dalpha_epspordt, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.epsilon_v, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.depsilon_vdt, memorySizeForParticles));
#endif

	cudaVerify(hipMalloc((void**)&p_device.x, memorySizeForTree));
	cudaVerify(hipMalloc((void**)&p_device.g_x, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.g_local_cellsize, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.vx, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dxdt, memorySizeForParticles));

#if DIM > 1
	cudaVerify(hipMalloc((void**)&p_device.y, memorySizeForTree));
	cudaVerify(hipMalloc((void**)&p_device.g_y, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.vy, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dydt, memorySizeForParticles));
    cudaVerify(hipMalloc((void**)&p_device.y0, memorySizeForTree));
    cudaVerify(hipMalloc((void**)&p_device.vy0, memorySizeForTree));
    cudaVerify(hipHostMalloc((void**)&p_host.vy0, memorySizeForTree));
#endif

    cudaVerify(hipMalloc((void**)&p_device.x0, memorySizeForTree));
    cudaVerify(hipMalloc((void**)&p_device.vx0, memorySizeForTree));
    cudaVerify(hipHostMalloc((void**)&p_host.vx0, memorySizeForTree));
#if DIM > 2
    cudaVerify(hipMalloc((void**)&p_device.z0, memorySizeForTree));
    cudaVerify(hipMalloc((void**)&p_device.vz0, memorySizeForTree));
    cudaVerify(hipHostMalloc((void**)&p_host.vz0, memorySizeForTree));
#endif

#if XSPH
	cudaVerify(hipMalloc((void**)&p_device.xsphvx, memorySizeForParticles));
#if DIM > 1
	cudaVerify(hipMalloc((void**)&p_device.xsphvy, memorySizeForParticles));
#endif
#endif
	cudaVerify(hipMalloc((void**)&p_device.ax, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.g_ax, memorySizeForParticles));

#if DIM > 1
	cudaVerify(hipMalloc((void**)&p_device.ay, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.g_ay, memorySizeForParticles));
#endif

	cudaVerify(hipMalloc((void**)&p_device.m, memorySizeForTree));
	cudaVerify(hipMalloc((void**)&p_device.h, memorySizeForParticles));

#if INTEGRATE_SML
	cudaVerify(hipMalloc((void**)&p_device.dhdt, memorySizeForParticles));
#endif

#if SML_CORRECTION
	cudaVerify(hipMalloc((void**)&p_device.sml_omega, memorySizeForParticles));
#endif

	cudaVerify(hipMalloc((void**)&p_device.rho, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.p, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.e, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.cs, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.depth, memorySizeForInteractions));
	cudaVerify(hipMalloc((void**)&p_device.noi, memorySizeForInteractions));
	cudaVerify(hipMalloc((void**)&p_device.materialId, memorySizeForInteractions));
	cudaVerify(hipMalloc((void**)&p_device.materialId0, memorySizeForInteractions));

#if MORE_OUTPUT
	cudaVerify(hipMalloc((void**)&p_device.p_min, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.p_max, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.rho_min, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.rho_max, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.e_min, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.e_max, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.cs_min, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.cs_max, memorySizeForParticles));
#endif

	cudaVerify(hipMalloc((void**)&interactions, memorySizeForInteractions*MAX_NUM_INTERACTIONS));
	cudaVerify(hipMalloc((void**)&childListd, memorySizeForChildren));
#if DIM > 2
	cudaVerify(hipMalloc((void**)&p_device.z, memorySizeForTree));
	cudaVerify(hipMalloc((void**)&p_device.g_z, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.dzdt, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.vz, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.az, memorySizeForParticles));
	cudaVerify(hipMalloc((void**)&p_device.g_az, memorySizeForParticles));
#if XSPH
	cudaVerify(hipMalloc((void**)&p_device.xsphvz, memorySizeForParticles));
#endif
#endif

	cudaVerify(hipMemset(p_device.ax, 0, memorySizeForParticles));
	cudaVerify(hipMemset(p_device.g_ax, 0, memorySizeForParticles));
#if DIM > 1
	cudaVerify(hipMemset(p_device.ay, 0, memorySizeForParticles));
	cudaVerify(hipMemset(p_device.g_ay, 0, memorySizeForParticles));
#endif
#if DIM > 2
	cudaVerify(hipMemset(p_device.az, 0, memorySizeForParticles));
	cudaVerify(hipMemset(p_device.g_az, 0, memorySizeForParticles));
#endif

    return rc;
}



int copy_particle_data_to_device()
{
    int rc = 0;

	if (param.verbose)
        fprintf(stdout, "\nCopying particle data to device...\n");

	cudaVerify(hipMemcpy(p_device.x0, p_host.x, memorySizeForTree, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.x, p_host.x, memorySizeForTree, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.vx, p_host.vx, memorySizeForParticles, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.vx0, p_host.vx0, memorySizeForParticles, hipMemcpyHostToDevice));
#if DIM > 1
	cudaVerify(hipMemcpy(p_device.y0, p_host.y, memorySizeForTree, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.y, p_host.y, memorySizeForTree, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.vy, p_host.vy, memorySizeForParticles, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.vy0, p_host.vy0, memorySizeForParticles, hipMemcpyHostToDevice));
#endif
#if DIM > 2
	cudaVerify(hipMemcpy(p_device.z0, p_host.z, memorySizeForTree, hipMemcpyHostToDevice));
#endif

#if GRAVITATING_POINT_MASSES
	cudaVerify(hipMemcpy(pointmass_device.x, pointmass_host.x, memorySizeForPointmasses, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(pointmass_device.vx, pointmass_host.vx, memorySizeForPointmasses, hipMemcpyHostToDevice));
# if DIM > 1
	cudaVerify(hipMemcpy(pointmass_device.y, pointmass_host.y, memorySizeForPointmasses, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(pointmass_device.vy, pointmass_host.vy, memorySizeForPointmasses, hipMemcpyHostToDevice));
#  if DIM > 2
	cudaVerify(hipMemcpy(pointmass_device.z, pointmass_host.z, memorySizeForPointmasses, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(pointmass_device.vz, pointmass_host.vz, memorySizeForPointmasses, hipMemcpyHostToDevice));
#  endif
# endif
	cudaVerify(hipMemcpy(pointmass_device.rmin, pointmass_host.rmin, memorySizeForPointmasses, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(pointmass_device.rmax, pointmass_host.rmax, memorySizeForPointmasses, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(pointmass_device.m, pointmass_host.m, memorySizeForPointmasses, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(pointmass_device.feels_particles, pointmass_host.feels_particles, integermemorySizeForPointmasses, hipMemcpyHostToDevice));
#endif

	cudaVerify(hipMemcpy(p_device.h, p_host.h, memorySizeForParticles, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.cs, p_host.cs, memorySizeForParticles, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.m, p_host.m, memorySizeForTree, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.rho, p_host.rho, memorySizeForParticles, hipMemcpyHostToDevice));
#if INTEGRATE_ENERGY
	cudaVerify(hipMemcpy(p_device.e, p_host.e, memorySizeForParticles, hipMemcpyHostToDevice));
#endif
#if SOLID
	cudaVerify(hipMemcpy(p_device.S, p_host.S, memorySizeForStress, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.ep, p_host.ep, memorySizeForParticles, hipMemcpyHostToDevice));
#endif
#if NAVIER_STOKES
	cudaVerify(hipMemcpy(p_device.Tshear, p_host.Tshear, memorySizeForStress, hipMemcpyHostToDevice));
#endif
#if PALPHA_POROSITY
	cudaVerify(hipMemcpy(p_device.alpha_jutzi, p_host.alpha_jutzi, memorySizeForParticles, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.alpha_jutzi_old, p_host.alpha_jutzi_old, memorySizeForParticles, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.p, p_host.p, memorySizeForParticles, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.pold, p_host.pold, memorySizeForParticles, hipMemcpyHostToDevice));
#endif
#if MORE_OUTPUT
    cudaVerify(hipMemcpy(p_device.p_min, p_host.p_min, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.p_max, p_host.p_max, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.rho_min, p_host.rho_min, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.rho_max, p_host.rho_max, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.e_min, p_host.e_min, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.e_max, p_host.e_max, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.cs_min, p_host.cs_min, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.cs_max, p_host.cs_max, memorySizeForParticles, hipMemcpyHostToDevice));
#endif
#if SIRONO_POROSITY
    cudaVerify(hipMemcpy(p_device.compressive_strength, p_host.compressive_strength, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.tensile_strength, p_host.tensile_strength, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.shear_strength, p_host.shear_strength, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.rho_0prime, p_host.rho_0prime, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.rho_c_plus, p_host.rho_c_plus, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.rho_c_minus, p_host.rho_c_minus, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.K, p_host.K, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.flag_rho_0prime, p_host.flag_rho_0prime, memorySizeForInteractions, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.flag_plastic, p_host.flag_plastic, memorySizeForInteractions, hipMemcpyHostToDevice));
#endif
#if EPSALPHA_POROSITY
    cudaVerify(hipMemcpy(p_device.alpha_epspor, p_host.alpha_epspor, memorySizeForParticles, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.epsilon_v, p_host.epsilon_v, memorySizeForParticles, hipMemcpyHostToDevice));
#endif
    cudaVerify(hipMemcpy(p_device.h0, p_host.h0, memorySizeForParticles, hipMemcpyHostToDevice));
#if JC_PLASTICITY
	cudaVerify(hipMemcpy(p_device.T, p_host.T, memorySizeForParticles, hipMemcpyHostToDevice));
#endif
#if FRAGMENTATION
	cudaVerify(hipMemcpy(p_device.d, p_host.d, memorySizeForParticles, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.numFlaws, p_host.numFlaws, memorySizeForInteractions, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.numActiveFlaws, p_host.numActiveFlaws, memorySizeForInteractions, hipMemcpyHostToDevice));
    cudaVerify(hipMemcpy(p_device.flaws, p_host.flaws, memorySizeForActivationThreshold, hipMemcpyHostToDevice));
# if PALPHA_POROSITY
    cudaVerify(hipMemcpy(p_device.damage_porjutzi, p_host.damage_porjutzi, memorySizeForParticles, hipMemcpyHostToDevice));
# endif
#endif
	cudaVerify(hipMemcpy(p_device.noi, p_host.noi, memorySizeForInteractions, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.materialId, p_host.materialId, memorySizeForInteractions, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.materialId0, p_host.materialId, memorySizeForInteractions, hipMemcpyHostToDevice));
#if DIM > 2
	cudaVerify(hipMemcpy(p_device.z, p_host.z, memorySizeForTree, hipMemcpyHostToDevice));
	cudaVerify(hipMemcpy(p_device.vz, p_host.vz, memorySizeForParticles, hipMemcpyHostToDevice));
#endif
	cudaVerify(hipMemset((void *) childListd, -1, memorySizeForChildren));

    return rc;
}



int free_memory()
{
    int rc = 0;

	/* free device memory */
	if (param.verbose)
        fprintf(stdout, "Freeing memory...\n");
	cudaVerify(hipFree(p_device.x));
	cudaVerify(hipFree(p_device.g_x));
	cudaVerify(hipFree(p_device.g_local_cellsize));
	cudaVerify(hipFree(p_device.depth));
	cudaVerify(hipFree(p_device.x0));
	cudaVerify(hipFree(p_device.dxdt));
	cudaVerify(hipFree(p_device.vx));
	cudaVerify(hipFree(p_device.vx0));
	cudaVerify(hipHostFree(p_host.vx0));
	cudaVerify(hipFree(p_device.ax));
	cudaVerify(hipFree(p_device.g_ax));
	cudaVerify(hipFree(p_device.m));

#if DIM > 1
	cudaVerify(hipFree(p_device.vy0));
	cudaVerify(hipHostFree(p_host.vy0));
#if DIM > 2
	cudaVerify(hipFree(p_device.vz0));
	cudaVerify(hipHostFree(p_host.vz0));
#endif
#endif
#if DIM > 1
	cudaVerify(hipFree(p_device.y));
	cudaVerify(hipFree(p_device.g_y));
	cudaVerify(hipFree(p_device.y0));
	cudaVerify(hipFree(p_device.vy));
	cudaVerify(hipFree(p_device.dydt));
	cudaVerify(hipFree(p_device.ay));
	cudaVerify(hipFree(p_device.g_ay));
#endif

#if GRAVITATING_POINT_MASSES
	cudaVerify(hipFree(pointmass_device.x));
	cudaVerify(hipFree(pointmass_device.vx));
	cudaVerify(hipFree(pointmass_device.ax));
	cudaVerify(hipFree(pointmass_device.feedback_ax));
# if DIM > 1
	cudaVerify(hipFree(pointmass_device.y));
	cudaVerify(hipFree(pointmass_device.vy));
	cudaVerify(hipFree(pointmass_device.ay));
	cudaVerify(hipFree(pointmass_device.feedback_ay));
#  if DIM > 2
	cudaVerify(hipFree(pointmass_device.z));
	cudaVerify(hipFree(pointmass_device.vz));
	cudaVerify(hipFree(pointmass_device.az));
	cudaVerify(hipFree(pointmass_device.feedback_az));
#  endif
# endif
	cudaVerify(hipFree(pointmass_device.m));
	cudaVerify(hipFree(pointmass_device.feels_particles));
	cudaVerify(hipFree(pointmass_device.rmin));
	cudaVerify(hipFree(pointmass_device.rmax));

	cudaVerify(hipHostFree(pointmass_host.x));
	cudaVerify(hipHostFree(pointmass_host.vx));
	cudaVerify(hipHostFree(pointmass_host.ax));
# if DIM > 1
	cudaVerify(hipHostFree(pointmass_host.y));
	cudaVerify(hipHostFree(pointmass_host.vy));
	cudaVerify(hipHostFree(pointmass_host.ay));
#  if DIM > 2
	cudaVerify(hipHostFree(pointmass_host.z));
	cudaVerify(hipHostFree(pointmass_host.vz));
	cudaVerify(hipHostFree(pointmass_host.az));
#  endif
# endif
	cudaVerify(hipHostFree(pointmass_host.m));
	cudaVerify(hipHostFree(pointmass_host.feels_particles));
	cudaVerify(hipHostFree(pointmass_host.rmin));
	cudaVerify(hipHostFree(pointmass_host.rmax));
#endif

#if XSPH
	cudaVerify(hipFree(p_device.xsphvx));
#if DIM > 1
	cudaVerify(hipFree(p_device.xsphvy));
#endif
#endif
	cudaVerify(hipFree(p_device.h));
	cudaVerify(hipFree(p_device.rho));
	cudaVerify(hipFree(p_device.p));
	cudaVerify(hipFree(p_device.e));
	cudaVerify(hipFree(p_device.cs));
	cudaVerify(hipFree(p_device.noi));
#if MORE_OUTPUT
	cudaVerify(hipFree(p_device.p_min));
    cudaVerify(hipFree(p_device.p_max));
    cudaVerify(hipFree(p_device.rho_min));
    cudaVerify(hipFree(p_device.rho_max));
	cudaVerify(hipFree(p_device.e_min));
    cudaVerify(hipFree(p_device.e_max));
    cudaVerify(hipFree(p_device.cs_min));
    cudaVerify(hipFree(p_device.cs_max));
#endif
#if ARTIFICIAL_VISCOSITY
	cudaVerify(hipFree(p_device.muijmax));
#endif
#if INVISCID_SPH
	cudaVerify(hipFree(p_device.beta));
	cudaVerify(hipFree(p_device.beta_old));
	cudaVerify(hipFree(p_device.divv_old));
#endif
	cudaVerify(hipFree(interactions));
	cudaVerify(hipFree(p_device.materialId));
	cudaVerify(hipFree(p_device.materialId0));
	cudaVerify(hipFree(childListd));
#if DIM > 2
	cudaVerify(hipFree(p_device.z));
	cudaVerify(hipFree(p_device.g_z));
	cudaVerify(hipFree(p_device.z0));
	cudaVerify(hipFree(p_device.dzdt));
	cudaVerify(hipFree(p_device.vz));
#if XSPH
	cudaVerify(hipFree(p_device.xsphvz));
#endif
	cudaVerify(hipFree(p_device.az));
	cudaVerify(hipFree(p_device.g_az));
#endif

#if TENSORIAL_CORRECTION
	cudaVerify(hipFree(p_device.tensorialCorrectionMatrix));
	//cudaVerify(hipFree(p_device.tensorialCorrectiondWdrr));
#endif

#if SHEPARD_CORRECTION
	cudaVerify(hipFree(p_device.shepard_correction));
#endif

#if INTEGRATE_ENERGY
	cudaVerify(hipHostFree(p_host.dedt));
	cudaVerify(hipFree(p_device.dedt));
#endif

	cudaVerify(hipHostFree(p_host.drhodt));
	cudaVerify(hipFree(p_device.drhodt));

#if INTEGRATE_SML
	cudaVerify(hipFree(p_device.dhdt));
#endif
#if SML_CORRECTION
	cudaVerify(hipFree(p_device.sml_omega));
#endif

#if NAVIER_STOKES
	cudaVerify(hipFree(p_device.Tshear));
	cudaVerify(hipHostFree(p_host.Tshear));
	cudaVerify(hipFree(p_device.eta));
#endif
#if SOLID
	cudaVerify(hipFree(p_device.S));
    cudaVerify(hipHostFree(p_host.ep));
	cudaVerify(hipFree(p_device.dSdt));
	cudaVerify(hipHostFree(p_host.S));
	cudaVerify(hipHostFree(p_host.dSdt));
	cudaVerify(hipFree(p_device.local_strain));
	cudaVerify(hipHostFree(p_host.local_strain));
    cudaVerify(hipFree(p_device.plastic_f));
	cudaVerify(hipFree(p_device.sigma));
    cudaVerify(hipFree(p_device.ep));
    cudaVerify(hipFree(p_device.edotp));
#endif
#if ARTIFICIAL_STRESS
	cudaVerify(hipFree(p_device.R));
#endif

#if JC_PLASTICITY
	cudaVerify(hipFree(p_device.T));
	cudaVerify(hipFree(p_device.dTdt));
	cudaVerify(hipFree(p_device.jc_f));
#endif

#if GHOST_BOUNDARIES
	cudaVerify(hipFree(p_device.real_partner));
#endif

#if FRAGMENTATION
	cudaVerify(hipHostFree(p_host.d));
	cudaVerify(hipFree(p_device.d));
	cudaVerify(hipFree(p_device.damage_total));
	cudaVerify(hipFree(p_device.dddt));
	cudaVerify(hipHostFree(p_host.dddt));
	cudaVerify(hipHostFree(p_host.numFlaws));
	cudaVerify(hipFree(p_device.numFlaws));
	cudaVerify(hipHostFree(p_host.numActiveFlaws));
	cudaVerify(hipFree(p_device.numActiveFlaws));
	cudaVerify(hipHostFree(p_host.flaws));
	cudaVerify(hipFree(p_device.flaws));
# if PALPHA_POROSITY
	cudaVerify(hipFree(p_device.damage_porjutzi));
	cudaVerify(hipFree(p_device.cs_old));
	cudaVerify(hipFree(p_device.ddamage_porjutzidt));
# endif
#endif


#if PALPHA_POROSITY
	cudaVerify(hipFree(p_device.alpha_jutzi));
	cudaVerify(hipFree(p_device.alpha_jutzi_old));
	cudaVerify(hipFree(p_device.pold));
	cudaVerify(hipFree(p_device.dalphadt));
	cudaVerify(hipFree(p_device.dalphadp));
	cudaVerify(hipFree(p_device.dp));
	cudaVerify(hipFree(p_device.dalphadrho));
	cudaVerify(hipFree(p_device.f));
	cudaVerify(hipFree(p_device.delpdelrho));
	cudaVerify(hipFree(p_device.delpdele));
#endif

#if SIRONO_POROSITY
    cudaVerify(hipFree(p_device.compressive_strength));
    cudaVerify(hipFree(p_device.tensile_strength));
    cudaVerify(hipFree(p_device.shear_strength));
    cudaVerify(hipFree(p_device.K));
    cudaVerify(hipFree(p_device.rho_0prime));
    cudaVerify(hipFree(p_device.rho_c_plus));
    cudaVerify(hipFree(p_device.rho_c_minus));
    cudaVerify(hipFree(p_device.flag_rho_0prime));
    cudaVerify(hipFree(p_device.flag_plastic));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(hipFree(p_device.alpha_epspor));
    cudaVerify(hipFree(p_device.dalpha_epspordt));
    cudaVerify(hipFree(p_device.epsilon_v));
    cudaVerify(hipFree(p_device.depsilon_vdt));
#endif

	cudaVerify(hipHostFree(p_host.x));
	cudaVerify(hipHostFree(p_host.vx));
	cudaVerify(hipHostFree(p_host.ax));
    cudaVerify(hipHostFree(p_host.g_ax));
#if DIM > 1
	cudaVerify(hipHostFree(p_host.y));
	cudaVerify(hipHostFree(p_host.vy));
	cudaVerify(hipHostFree(p_host.ay));
    cudaVerify(hipHostFree(p_host.g_ay));
#endif
	cudaVerify(hipHostFree(p_host.m));
	cudaVerify(hipHostFree(p_host.h));
	cudaVerify(hipHostFree(p_host.rho));
	cudaVerify(hipHostFree(p_host.p));
	cudaVerify(hipHostFree(p_host.e));
	cudaVerify(hipHostFree(p_host.cs));
	cudaVerify(hipHostFree(p_host.noi));
	cudaVerify(hipHostFree(interactions_host));
	cudaVerify(hipHostFree(p_host.depth));
	cudaVerify(hipHostFree(p_host.materialId));
	cudaVerify(hipHostFree(childList_host));
#if MORE_OUTPUT
	cudaVerify(hipHostFree(p_host.p_min));
	cudaVerify(hipHostFree(p_host.p_max));
	cudaVerify(hipHostFree(p_host.rho_min));
	cudaVerify(hipHostFree(p_host.rho_max));
	cudaVerify(hipHostFree(p_host.e_min));
	cudaVerify(hipHostFree(p_host.e_max));
	cudaVerify(hipHostFree(p_host.cs_min));
	cudaVerify(hipHostFree(p_host.cs_max));
#endif
#if INVISCID_SPH
	cudaVerify(hipHostFree(p_host.beta));
	cudaVerify(hipHostFree(p_host.beta_old));
	cudaVerify(hipHostFree(p_host.divv_old));
#endif
#if PALPHA_POROSITY
	cudaVerify(hipHostFree(p_host.alpha_jutzi));
	cudaVerify(hipHostFree(p_host.alpha_jutzi_old));
	cudaVerify(hipHostFree(p_host.dalphadt));
	cudaVerify(hipHostFree(p_host.pold));
# if FRAGMENTATION
    cudaVerify(hipHostFree(p_host.damage_porjutzi));
    cudaVerify(hipHostFree(p_host.ddamage_porjutzidt));
# endif
#endif

#if SIRONO_POROSITY
    cudaVerify(hipHostFree(p_host.compressive_strength));
    cudaVerify(hipHostFree(p_host.tensile_strength));
    cudaVerify(hipHostFree(p_host.shear_strength));
    cudaVerify(hipHostFree(p_host.rho_0prime));
    cudaVerify(hipHostFree(p_host.rho_c_plus));
    cudaVerify(hipHostFree(p_host.rho_c_minus));
    cudaVerify(hipHostFree(p_host.K));
    cudaVerify(hipHostFree(p_host.flag_rho_0prime));
    cudaVerify(hipHostFree(p_host.flag_plastic));
#endif

#if EPSALPHA_POROSITY
    cudaVerify(hipHostFree(p_host.alpha_epspor));
    cudaVerify(hipHostFree(p_host.epsilon_v));
#endif

#if JC_PLASTICITY
	cudaVerify(hipHostFree(p_host.T));
#endif
#if DIM > 2
	cudaVerify(hipHostFree(p_host.z));
	cudaVerify(hipHostFree(p_host.vz));
	cudaVerify(hipHostFree(p_host.az));
    cudaVerify(hipHostFree(p_host.g_az));
#endif

    free_aneos_memory();

    return rc;
}
